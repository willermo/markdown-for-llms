"""`mdf bench run`: measured conversions of the reference set through the engine, and their resumption.

A run converts each selected document on each selected route of its format through
`adapters.engine.convert` (the engine, plan r2 §7), scores the outputs (`report`) and writes
`report.json` and `report.md`. Everything is under the run directory, locally, with document text:

    run.lock                   the lock of the run (delta r3 Δ2)
    state.json                 the state, written atomically with fsync after every change
    attempts/<attempt_id>/out  the output of the engine, never removed
    attempts/<attempt_id>/results.json   the outcome of every check, by id
    work/                      the private directories of the engine
    report.json, report.md     numbers and closed codes only

Attempts (Δ1). A key is (document, route, variant, repetition k); the variant is the canonical
JSON of the options that change the conversion (`{"ocr": …}` for Marker, `{}` for the other local
routes) and `v8` the first 8 hexadecimal digits of its SHA-256. Each key has generations
g1, g2, …; `attempt_id` = `<run8>-<doc>-<route>-<v8>-<k>-g<n>` names the output directory, the
work directory of the engine and the record in the state. Two variants with the same `v8` in a run
are `attempt_id_collision` before any conversion.

Order (C-ORDINE of the review of the delta). A new run validates its options, the manifest, the
exclusions and the selection first, so a refusal leaves no directory; then it creates the run
directory (`run_exists` if it is there), takes its lock and writes the first state. A resumed run
(`--resume`, every option from the state, Δ7) takes the lock of the run before it reads the state;
a lock held by another process is `run_busy`, and nothing is touched. Before either, the bench
takes a shared lock on the development venv (`venv_busy` while the full check holds it, §8) and
refuses to start while a process of the user other than a bench and its relatives runs a program
of the venv (`venv_in_use`, Δ10: API and worker are stopped by procedure).

Caps (C-ADOZIONE). The execution caps of each document (the limits of the product,
`Settings().engine_limits()`, with the `limits` of the manifest) are frozen in the state with the
request of each generation before the call; `engine.json.limits` is something else, the declared
limitations of the route, kept as a diagnostic of the result.

Resumption (§7, Δ8). The identity (package and its `RECORD`, toolchain image, weights, manifest
and exclusions) must be unchanged (`resume_identity_changed`). A key whose last generation is done
is never converted again. A generation that was started or interrupted is adopted when its
`engine.json` (read up to 1 MiB) has the same attempt, source SHA-256 and requested options as the
state, components that agree with the identity, an inventory whose files are regular files of
`out` with the declared bytes and SHA-256, and status `complete`, or `failed` with a code other
than `interrupted`, `datalab_cap` and the stops before the wake of a server (`SERVER_STOPS`, MFL033
G-B2); otherwise it is abandoned, kept, and the key is converted in a new generation. On
`marker-server` the budget and the session of an attempt are in its request before the state saves
it, so the state holds the options its `engine.json` records.

Datalab (plan r2 §12, delta r3 Δ5-Δ7). Only with `--route datalab`, the modes, the confirmed
transit, the two caps and the key file; never on a personal set (`datalab_refused_personal`,
before the key file is opened) and never on D20 (it would send the PDF of D14). The key is read
only with `engine.datalab.read_key(key_file, environ={})`, never kept in the state or written
anywhere. The variant of a Datalab key is `{"datalab_mode": …}`; its keys come after the local
ones, the cheapest first (D16 `accurate` last), one at a time, `DATALAB_SPACING` seconds apart.
Before each call the execution cap `cap_usd` of the engine is the smallest of what the ledger
cap and the run cap leave and of the estimate of the document, frozen in the request of the
generation; at the event `datalab-send` (before the POST) the observer checks the caps again
under the lock of the ledger and appends the reservation (`spend.Ledger`), or raises
`EngineError(datalab_cap)` and nothing is sent. A Datalab result is settled in one place,
`Run.settle`, after the call and after an adoption alike (B-SPESA of the review of the final
diff): the completion closes the reservation (`spend.completion_billed`), and only then is the
attempt done; then its billed cost is checked against its estimate. A cap that stops a send, a
ledger that cannot be read or written, and a billed cost over its estimate stop the run (exit 1)
before any other send, with the state (`status: "stopped"`, `stop`: the code) and the report
saved. A stop for a billed cost over its estimate stays: it is written in the state by the same
save that makes the attempt done, before its line, and a resumption also finds it in the ledger
(a completion billed over the estimate of its reservation), so every later resumption stops again
with it and sends nothing, until the maintainer has looked at the cost. On a resumption a Datalab
generation is never sent again once its reservation is in the ledger: without an adoptable result
it is `datalab_unresolved`; `--retry-unaccepted` converts again only an attempt refused before its
acceptance (`datalab_rejected` without a `request_id`). Each attempt that a resumption adopts or
closes as `datalab_unresolved` prints its line.

Report again (`rebuild_report`, `mdf bench report RUN_DIR`). The report of a finished run is
written again from its saved outputs, with the scoring of the installed package, after its
identity, its check sets and its results are verified; nothing is converted or sent. With
`--report-dir DIR` it is read-only on the run (MFL044 (d)): no lock and no write in the run
directory, the report and the outcome of each check written to DIR (new or empty, outside the
run), the state read again at the end (`report_run_changed` when it moved).

Paths (MFL044 (d)). The state of schema 2 records the paths of a run so that the run can move: a
manifest and exclusions given by hand are copied into `RUN/inputs/` (their SHA-256 in the state),
the runs directory is relative to the run, and the cache and the model root are hints, replaced by
`--cache` and `--model-root` on a resumption (`--cache`, `--manifest` and `--exclusions` on a
report) and verified by what the state recorded (the SHA-256 of inputs, check sets, manifest and
exclusions; the identity of the weights). A path that is missing without its option is
`run_path_missing`. A state of schema 1 stays readable: its absolute paths are the hints, and its
runs directory, when gone, is the parent of the run. A run with Datalab is resumed only where it
was created, next to its ledger (`run_path_missing` otherwise); its report reads the ledger of its
runs directory.
"""
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
from decimal import Decimal, InvalidOperation
import resource
import secrets
import stat
import sys
import time

from . import exclusions as excl_mod, manifest as manifest_mod, spend
from .adapters import engine as engine_adapter
from .cache import Cache, canonical, sha256_file, write_atomic
from .codes import BenchError, Code

STATE_SCHEMA = "mdf-bench-run/2"
STATE_SCHEMAS = ("mdf-bench-run/1", STATE_SCHEMA)
STATE, LOCK = "state.json", "run.lock"
INPUTS = "inputs"                   # copies of a manifest and exclusions given by hand (schema 2)
STATE_KEYS = ("run_id", "selection", "manifest", "exclusions", "cache", "runs_dir", "model_root", "caps", "identity",
              "environment", "documents", "keys", "attempts", "generations", "executions")
MAX_STATE_BYTES = 64 * 1048576
MAX_RESULT_BYTES = 1048576          # engine.json read for an adoption
MAX_REPEAT = 10
AUTO = object()                     # the venv of this interpreter
DATALAB = "datalab"
MARKER_SERVER = "marker-server"
SERVER_STOPS = ("server_cap", "session_orphaned", "server_not_armed", "server_billed_over_bound")
WEIGHTED_ROUTES = frozenset({"marker", "marker-server"})   # routes whose child reads the weights of the model root
DATALAB_SPACING = 5.0               # seconds between two Datalab attempts (plan r2 §12)
# Datalab is never asked for these documents (Davide, 2026-11-16 10:36: D20 sends the PDF of D14)
DATALAB_EXCLUDED = {"D20": "same_pdf_as_d14"}
DATALAB_WARNING = ("Datalab: the selected documents of the public set leave this machine for Datalab "
                   "(https://www.datalab.to, US region). Training on the data and retention are on for the "
                   "team, and the API can neither read nor change them; the DELETE of each result may not "
                   "remove the copy kept for training. Personal documents are never sent.")
TERMINAL = ("done", "unresolved")   # statuses of a generation whose key is never converted again
# a stop that a resumption keeps until the maintainer decides (the run is not resumed past it)
STICKY_STOPS = ("datalab_billed_over_estimate", "server_billed_over_bound")


class Stop(Exception):
    """A stop of the run decided by the bench (a cap, the ledger, a billed cost over its estimate)."""

    def __init__(self, code):
        super().__init__(code.value)
        self.code = code


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- selection and identities of the attempts -------------------------------------------------

def local_routes(fmt):
    """The routes of a default selection: the matrix without the routes asked only by name
    (`routes.EXPLICIT`: Datalab and the inference server)."""
    from ..engine.routes import EXPLICIT, MATRIX
    default, alternatives = MATRIX[fmt]
    return [r for r in (default, *alternatives) if r not in EXPLICIT]


def parse_routes(text):
    """The routes asked with `--route` (None: every local route of each format)."""
    from ..engine.routes import CONVERTERS, DATALAB
    if text is None:
        return None
    wanted = list(dict.fromkeys(x.strip() for x in text.split(",") if x.strip()))
    if not wanted or any(r not in CONVERTERS for r in wanted):
        raise BenchError(Code.ROUTE_INVALID, "route")
    return wanted


def parse_modes(text):
    """The Datalab modes asked with `--datalab-mode`, in the order of `routes.DATALAB_MODES`."""
    from ..engine.routes import DATALAB_MODES
    wanted = [x.strip() for x in (text or "").split(",") if x.strip()]
    if not wanted:
        raise BenchError(Code.INVALID_SETTING, "--datalab-mode with --route datalab")
    if any(m not in DATALAB_MODES for m in wanted):
        raise BenchError(Code.ROUTE_OPTION_UNSUPPORTED, "datalab mode")
    return [m for m in DATALAB_MODES if m in wanted]


def parse_server_modes(text):
    """The modes asked with `--server-mode`, in the order of `routes.SERVER_MODES`."""
    from ..engine.routes import SERVER_MODES
    wanted = [x.strip() for x in (text or "").split(",") if x.strip()]
    if not wanted:
        raise BenchError(Code.INVALID_SETTING, "--server-mode with --route marker-server")
    if any(m not in SERVER_MODES for m in wanted):
        raise BenchError(Code.ROUTE_OPTION_UNSUPPORTED, "server mode")
    return [m for m in SERVER_MODES if m in wanted]


def parse_usd(value, what):
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        amount = None
    if amount is None or not amount.is_finite() or amount < 0 or amount > 1000:
        raise BenchError(Code.INVALID_SETTING, what + ": an amount in USD from 0 to 1000")
    return amount


def variant(route, ocr, mode=None, server_mode=None):
    from ..engine.routes import MARKER
    if route == MARKER:
        return {"ocr": ocr}
    if route == MARKER_SERVER:
        return {"ocr": ocr, "server_mode": server_mode}
    if route == DATALAB:
        return {"datalab_mode": mode}
    return {}


def variant_sha8(value):
    return hashlib.sha256(canonical(value)).hexdigest()[:8]


def key_name(doc_id, route, v8, k):
    return f"{doc_id}|{route}|{v8}|{k}"


def attempt_id(run8, doc_id, route, v8, k, g):
    return f"{run8}-{doc_id}-{route}-{v8}-{k}-g{g}"


def datalab_estimate(doc, mode):
    """The estimate of the engine for a document of the public set, from its pages (verified)."""
    from ..engine.datalab import estimate_cents
    return estimate_cents(doc["pages"], mode)


def plan_keys(docs, routes, ocr, repeat, modes=None, server_modes=None):
    """The keys of a selection: the local ones in order (repetition, document, route; with
    `marker-server` asked, its modes after the local routes of each document), then the Datalab
    ones, the cheapest first (repetition, estimate, document, mode); refuses colliding variants."""
    from ..engine.routes import DATALAB_MODES, MATRIX
    keys, remote, seen = [], [], {}

    def add(target, doc, route, v, k, order=None):
        v8 = variant_sha8(v)
        if seen.setdefault((route, v8), v) != v:
            raise BenchError(Code.ATTEMPT_ID_COLLISION, f"{route} {v8}")
        target.append((order, {"document": doc["id"], "route": route, "variant": v, "v8": v8, "k": k}))
    for k in range(1, repeat + 1):
        for doc in docs:
            for route in local_routes(doc["format"]):
                if routes is not None and route not in routes:
                    continue
                add(keys, doc, route, variant(route, ocr), k)
            if routes is not None and MARKER_SERVER in routes and MARKER_SERVER in MATRIX[doc["format"]][1]:
                for mode in server_modes or ():
                    add(keys, doc, MARKER_SERVER, variant(MARKER_SERVER, ocr, server_mode=mode), k)
            if routes is not None and DATALAB in routes and DATALAB in MATRIX[doc["format"]][1] \
                    and doc["id"] not in DATALAB_EXCLUDED:
                for mode in modes:
                    add(remote, doc, DATALAB, variant(DATALAB, ocr, mode), k,
                        (k, datalab_estimate(doc, mode), doc["id"], DATALAB_MODES.index(mode)))
    keys = [key for _, key in keys] + [key for _, key in sorted(remote, key=lambda x: x[0])]
    if not keys:
        raise BenchError(Code.INVALID_SETTING, "no route of the selection")
    return keys


def selection_keys(docs, sel):
    """The keys of the saved selection `sel` of a run, for `resume` and `rebuild_report` alike (MFL033
    G-B1: the report of a run with `marker-server` needs its modes, like the resumption)."""
    return plan_keys(docs, sel["routes"], sel["ocr"], sel["repeat"], sel.get("datalab_modes"), sel.get("server_modes"))


def caps_of(doc):
    """The execution caps of a document: the limits of the product and the `limits` of the manifest."""
    from ..settings import Settings
    return {**Settings().engine_limits(), **doc.get("limits", {})}


def input_name(doc):
    """The name the engine records for the input: the document id and the canonical extension."""
    from ..engine.formats import FORMATS
    return f"{doc['id']}.{FORMATS[doc['format']][1][0]}"


def expected_requested(request):
    """`options.requested` as `engine.core.convert` records it for a request."""
    requested = {k: request.get(k) for k in ("route", "ocr", "lang")}
    if request.get("datalab") is not None:
        requested["datalab"] = {k: request["datalab"].get(k) for k in ("mode", "transit_confirmed", "cap_usd")}
    if isinstance(request.get("server"), dict):
        requested["server"] = {k: request["server"].get(k) for k in ("name", "mode", "transit_confirmed", "budget_cents")}
        requested["server"]["arm_id"] = (request["server"].get("session") or {}).get("arm_id")
    return requested


# --- locks and the venv -----------------------------------------------------------------------

def venv_root():
    return Path(sys.prefix) if sys.prefix != sys.base_prefix else None


def lock_venv(venv):
    """A shared lock on the venv directory: the full check holds it exclusively (`venv_busy`)."""
    if venv is None:
        return None
    fd = os.open(venv, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise BenchError(Code.VENV_BUSY, "the full check holds the venv") from None
    return fd


def _proc_table():
    """{pid: (ppid, argv, cwd)} of the processes of this user that can be read."""
    table, uid = {}, os.getuid()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            with open(f"/proc/{entry.name}/stat", "rb") as fh:
                ppid = int(fh.read().rsplit(b")", 1)[1].split()[1])
            with open(f"/proc/{entry.name}/cmdline", "rb") as fh:
                argv = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
            try:
                cwd = os.readlink(f"/proc/{entry.name}/cwd")
            except OSError:
                cwd = None
        except (OSError, ValueError, IndexError):
            continue
        table[int(entry.name)] = (ppid, argv, cwd)
    return table


def _is_bench(argv):
    for i, arg in enumerate(argv[:-1]):
        if argv[i + 1] == "bench" and (arg == "markdown_for_llms.cli" or os.path.basename(arg) == "mdf"):
            return True
    return False


def _runs_venv(argv, cwd, venv_bin):
    """True when the program (or the script an interpreter runs by its shebang) is in `venv/bin`."""
    for arg in argv[:2]:
        if "/" not in arg or (cwd is None and not arg.startswith("/")):
            continue
        path = os.path.normpath(os.path.join(cwd or "/", arg))
        if os.path.realpath(os.path.dirname(path)) == venv_bin:
            return True
    return False


def venv_users(venv, own=None):
    """PIDs of the processes of the user that run a program of the venv, except this process, its
    ancestors and descendants, and the other bench processes and their descendants (Δ10, C-ORDINE:
    two runs of one directory are kept apart by the lock of the run, `run_busy`)."""
    own = os.getpid() if own is None else own
    venv_bin = os.path.realpath(Path(venv) / "bin")
    table = _proc_table()
    children = {}
    for pid, (ppid, _, _) in table.items():
        children.setdefault(ppid, []).append(pid)

    def subtree(root):
        out, stack = set(), [root]
        while stack:
            p = stack.pop()
            if p not in out:
                out.add(p)
                stack.extend(children.get(p, ()))
        return out
    excluded = subtree(own)
    p = own
    while p in table and p not in (0, 1) and len(excluded) < 100000:
        p = table[p][0]
        excluded.add(p)
    for pid, (_, argv, _) in table.items():
        if _is_bench(argv):
            excluded |= subtree(pid)
    return sorted(pid for pid, (_, argv, cwd) in table.items()
                  if pid not in excluded and _runs_venv(argv, cwd, venv_bin))


def check_venv_users(venv):
    if venv is not None and venv_users(venv):
        raise BenchError(Code.VENV_IN_USE, "a process runs a program of the development venv")


def lock_new(run_dir):
    fd = os.open(run_dir / LOCK, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise BenchError(Code.RUN_BUSY, "lock of the run") from None
    return fd


def lock_existing(run_dir):
    try:
        fd = os.open(Path(run_dir) / LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        raise BenchError(Code.RESUME_STATE_MISSING, "no lock file of a run") from None
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        os.close(fd)
        raise BenchError(Code.RESUME_STATE_MISSING, "the lock is not a regular file of the user")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise BenchError(Code.RUN_BUSY, "another process holds the run") from None
    return fd


# --- state ------------------------------------------------------------------------------------

def save_state(run_dir, state):
    try:
        write_atomic(Path(run_dir) / STATE, json.dumps(state, indent=1, sort_keys=True).encode("utf-8"))
    except OSError as exc:
        raise BenchError(Code.STATE_UNWRITABLE, errno.errorcode.get(exc.errno, "OSError")) from None


def load_state(run_dir):
    try:
        fd = os.open(Path(run_dir) / STATE, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        raise BenchError(Code.RESUME_STATE_MISSING, "no state of a run") from None
    with os.fdopen(fd, "rb") as fh:
        info = os.fstat(fh.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STATE_BYTES:
            raise BenchError(Code.RESUME_STATE_INVALID, "state")
        data = fh.read(MAX_STATE_BYTES + 1)
    try:
        state = json.loads(data)
    except ValueError:
        raise BenchError(Code.RESUME_STATE_INVALID, "state JSON") from None
    if not isinstance(state, dict) or state.get("schema") not in STATE_SCHEMAS or not set(STATE_KEYS) <= set(state) \
            or not isinstance(state["identity"], dict) or not isinstance(state["attempts"], dict):
        raise BenchError(Code.RESUME_STATE_INVALID, "state schema")
    return state


def run_paths(run_dir, state, cache=None, model_root=None, manifest=None, exclusions=None):
    """The paths of a run where it is now (MFL044 (d), see the module docstring): {manifest,
    exclusions (None for the packaged files), cache, model_root, runs_dir, in_place}. An option
    replaces the recorded hint; a recorded or given path that is missing is `run_path_missing`
    (the cache and the model root are checked when used, `need`)."""
    run_dir = Path(run_dir).absolute()
    v2 = state.get("schema") == STATE_SCHEMA

    def recorded(entry):
        path = entry.get("path") if isinstance(entry, dict) else None
        if path is None:
            return None
        return run_dir / path if v2 else Path(path)

    def file(given, entry, what):
        path = Path(given).absolute() if given is not None else recorded(entry)
        if path is not None and not path.is_file():
            raise BenchError(Code.RUN_PATH_MISSING, what)
        return path

    if v2:
        runs_dir = Path(os.path.normpath(run_dir / state["runs_dir"]))
        in_place = (state.get("origin") or {}).get("run_dir") == str(run_dir)
    else:
        recorded_runs = Path(state["runs_dir"])
        runs_dir = recorded_runs if recorded_runs.is_dir() else run_dir.parent
        in_place = recorded_runs.is_dir() and run_dir.is_relative_to(recorded_runs)
    return {"manifest": file(manifest, state["manifest"], "manifest"),
            "exclusions": file(exclusions, state["exclusions"], "exclusions"),
            "cache": Path(cache if cache is not None else state["cache"]).absolute(),
            "model_root": Path(model_root if model_root is not None else state["model_root"]).absolute(),
            "runs_dir": runs_dir, "in_place": in_place}


def need(paths, *names):
    """The directories `names` of `paths` must exist (`run_path_missing`)."""
    for name in names:
        if not Path(paths[name]).is_dir():
            raise BenchError(Code.RUN_PATH_MISSING, name)


def copy_input(run_dir, given, name, sha256):
    """A manifest or exclusions file given by hand, copied into RUN/inputs/ with the SHA-256 that
    was read (otherwise the file changed meanwhile: `manifest_invalid`); its path in the state."""
    if given is None:
        return None
    data = Path(given).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise BenchError(Code.MANIFEST_INVALID, name + " changed while the run started")
    (run_dir / INPUTS).mkdir(mode=0o700, exist_ok=True)
    write_atomic(run_dir / INPUTS / name, data)
    return INPUTS + "/" + name


def _manifest(path):
    try:
        return manifest_mod.load(path)
    except OSError:
        raise BenchError(Code.MANIFEST_INVALID, "manifest unreadable") from None


def _exclusions(path):
    """(data, SHA-256 of the file) of the exclusions."""
    from importlib import resources
    try:
        raw = (resources.files("markdown_for_llms.bench").joinpath("data/exclusions.json").read_bytes() if path is None
               else Path(path).read_bytes())
    except OSError:
        raise BenchError(Code.MANIFEST_INVALID, "exclusions unreadable") from None
    return excl_mod.load(path), hashlib.sha256(raw).hexdigest()


def _identity_differences(old, new):
    keys = (("package", "version"), ("package", "record_sha256"), ("toolchain", "content_sha256"),
            ("toolchain", "image_id"), ("toolchain", "code"), ("weights", "manifest_sha256"), ("weights", "code"))
    return [f"{a}.{b}" for a, b in keys if (old.get(a) or {}).get(b) != (new.get(a) or {}).get(b)]


# --- adoption (Δ8, C-ADOZIONE) ----------------------------------------------------------------

def read_result(out_dir):
    """The parsed `engine.json` of an output (None when absent, unreadable, too long or not JSON)."""
    try:
        fd = os.open(Path(out_dir) / "engine.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as fh:
        info = os.fstat(fh.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RESULT_BYTES:
            return None
        data = fh.read(MAX_RESULT_BYTES + 1)
    try:
        result = json.loads(data)
    except ValueError:
        return None
    return result if isinstance(result, dict) and len(data) <= MAX_RESULT_BYTES else None


def inventory_ok(out_dir, inventory):
    """Every listed file is a regular file inside `out`, with the declared bytes and SHA-256."""
    out_dir = Path(out_dir)
    if not isinstance(inventory, list) or not any(isinstance(i, dict) and i.get("path") == "document.md"
                                                    for i in inventory):
        return False
    for item in inventory:
        rel = item.get("path") if isinstance(item, dict) else None
        if not isinstance(rel, str) or not rel or rel.startswith("/") or any(p in ("", ".", "..") for p in rel.split("/")):
            return False
        current = out_dir
        for part in rel.split("/"):
            current = current / part
            try:
                info = os.lstat(current)
            except OSError:
                return False
            if stat.S_ISLNK(info.st_mode):
                return False
        if not stat.S_ISREG(info.st_mode) or info.st_size != item.get("bytes") or sha256_file(current) != item.get("sha256"):
            return False
    return True


def components_ok(components, identity):
    """The components of a result agree with the identity of the run."""
    if not isinstance(components, dict):
        return False
    if (components.get("package") or {}).get("markdown-for-llms") != identity["package"]["version"]:
        return False
    tc = components.get("toolchain")
    if tc is not None and (not isinstance(tc, dict) or tc.get("content_sha256") != identity["toolchain"]["content_sha256"]
                           or tc.get("image_id") != identity["toolchain"]["image_id"]):
        return False
    w = components.get("weights")
    if w is not None and (not isinstance(w, dict) or w.get("manifest_sha256") != identity["weights"]["manifest_sha256"]):
        return False
    return True


def _belongs(attempt, r):
    """The result `r` (a parsed `engine.json`) is of this attempt: schema, attempt, input and options."""
    from ..engine.contract import SCHEMA
    if r is None or r.get("schema") != SCHEMA or r.get("attempt") != attempt["id"]:
        return False
    if not isinstance(r.get("source"), dict) or r["source"].get("sha256") != attempt["input_sha256"]:
        return False
    return isinstance(r.get("options"), dict) and r["options"].get("requested") == expected_requested(attempt["request"])


def adoptable(attempt, out_dir, identity):
    """True when the output of a started or interrupted generation can stand as its result."""
    r = read_result(out_dir)
    if not _belongs(attempt, r):
        return False
    status = r.get("status")
    if status == "complete":
        return components_ok(r.get("components"), identity) and inventory_ok(out_dir, r.get("inventory"))
    if status == "failed":
        code = (r.get("error") or {}).get("code") if isinstance(r.get("error"), dict) else None
        # an interruption is no outcome; a Datalab cap sent nothing, and a stop of the session or of a cap
        # came before the server woke (MFL033 G-B2): each is converted again
        if not isinstance(code, str) or code in (Code.INTERRUPTED.value, Code.DATALAB_CAP.value, *SERVER_STOPS):
            return False
        return r.get("components") is None or components_ok(r.get("components"), identity)
    return False


def result_verified(attempt, out_dir, identity):
    """For `rebuild_report`: the result of a finished generation is still the one the state recorded:
    its `engine.json` has the SHA-256 recorded when the generation ended or was adopted, belongs to it
    (`_belongs`), has components that agree with the identity of the run and, when complete, the files
    of its inventory (`inventory_ok`). A generation that recorded no result has none on disk either."""
    recorded = (attempt.get("result") or {}).get("engine_sha256")
    path = Path(out_dir) / "engine.json"
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return recorded is None
    if recorded is None or not stat.S_ISREG(info.st_mode) or sha256_file(path) != recorded:
        return False
    r = read_result(out_dir)
    if not _belongs(attempt, r):
        return False
    if r.get("status") == "complete":
        return components_ok(r.get("components"), identity) and inventory_ok(out_dir, r.get("inventory"))
    return r.get("status") == "failed" and (r.get("components") is None or components_ok(r.get("components"), identity))


def read_checkset(path):
    """A check set file (no symbolic link) whose checks have the SHA-256 it declares, else None."""
    from .checks import SCHEMA as CHECKS_SCHEMA
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            return None
        data = fh.read()
    try:
        body = json.loads(data)
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("schema") != CHECKS_SCHEMA or not isinstance(body.get("checks"), list):
        return None
    return body if body.get("checks_sha256") == hashlib.sha256(canonical(body["checks"])).hexdigest() else None


def checks_index(checks_dir):
    """{(document, checks_sha256): body} of the check sets of a cache made by the generator and the
    normalizer of this package (the cache keeps the sets of earlier versions too, often with the same
    checks); None for a pair that two such bodies claim with different counts or kinds (then no
    report is written from it)."""
    from . import GENERATOR_VERSION, NORMALIZER_VERSION

    def relevant(body):
        return canonical({k: body.get(k) for k in ("twin_kind", "not_determinable", "feedback", "rules")})
    index = {}
    try:
        names = sorted(os.listdir(checks_dir))
    except OSError:
        return index
    for name in names:
        body = read_checkset(Path(checks_dir) / name) if name.endswith(".json") else None
        if body is None or (body.get("generator"), body.get("normalizer")) != (GENERATOR_VERSION, NORMALIZER_VERSION):
            continue
        pair = (body.get("document"), body["checks_sha256"])
        if pair not in index:
            index[pair] = body
        elif index[pair] is not None and relevant(index[pair]) != relevant(body):
            index[pair] = None
    return index


def result_record(out_dir):
    r = read_result(out_dir) or {}
    error = r.get("error") if isinstance(r.get("error"), dict) else {}
    path = Path(out_dir) / "engine.json"
    return {"status": r.get("status"), "code": error.get("code"),
            "engine_sha256": sha256_file(path) if path.is_file() and not path.is_symlink() else None}


# --- environment ------------------------------------------------------------------------------

def environment(identity):
    import platform
    env = {"cpu": None, "threads": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)), "ram_gib": None,
           "kernel": platform.release(), "python": platform.python_version(),
           "docker": identity["toolchain"].get("engine")}
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                env["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text(errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                env["ram_gib"] = round(int(line.split()[1]) / 1048576, 1)
                break
    except (OSError, ValueError):
        pass
    return env


# --- the run ----------------------------------------------------------------------------------

class DatalabSend:
    """The observer of a Datalab attempt (Δ6). At the event `datalab-send`, before the POST, the caps
    are checked again under the lock of the ledger and the reservation is appended with fsync before
    the observer returns; when a cap would be exceeded, or the ledger cannot be read or written,
    nothing is appended and `EngineError(datalab_cap)` stops the attempt before any byte leaves (any
    other exception would become `engine_failed` in the engine). `failure` keeps the reason."""

    def __init__(self, run, attempt, pages, estimate):
        self.run, self.attempt, self.pages, self.estimate = run, attempt, pages, estimate
        self.reserved, self.failure = False, None

    def __call__(self, event):
        if event.get("event") == "step" and event.get("step") == "datalab-send":
            self.reserve()
        return None

    def reserve(self):
        state, a = self.run.state, self.attempt
        caps = state["datalab"]
        failure = detail = None
        try:
            with self.run.ledger.locked() as ledger:
                total, run_total = spend.totals(ledger.entries(), state["run_id"])
                if total + self.estimate > Decimal(caps["spend_cap_usd"]) * 100:
                    failure, detail = Code.DATALAB_CAP, "the spend cap of the ledger"
                elif run_total + self.estimate > Decimal(caps["max_cost_usd"]) * 100:
                    failure, detail = Code.DATALAB_CAP, "the cost cap of the run"
                else:
                    ledger.append({"kind": "reserve", "run_id": state["run_id"], "attempt_id": a["id"],
                                   "document": a["document"], "route": a["route"], "variant": a["variant"],
                                   "v8": a["v8"], "k": a["k"], "g": a["g"], "pages": self.pages,
                                   "estimate_cents": str(self.estimate), "at": spend.now()})
                    self.reserved = True
        except BenchError as exc:            # the ledger does not reconcile, or is not a private file
            failure, detail = exc.code, "spend ledger inconsistent"
        except OSError:
            failure, detail = Code.SPEND_LEDGER_UNWRITABLE, "spend ledger not writable"
        if failure is not None:
            self.failure = failure
            raise BenchError(Code.DATALAB_CAP, detail)


class Run:
    def __init__(self, run_dir, state, manifest, exclusions, docs, keys, key=None, paths=None, readonly=False,
                 server=None):
        self.dir, self.state, self.manifest, self.exclusions = Path(run_dir), state, manifest, exclusions
        self.docs, self.keys = docs, keys
        self.paths = paths if paths is not None else run_paths(run_dir, state)
        self.readonly = readonly            # `mdf bench report --report-dir`: the ledger read without its lock
        self.cache = Cache(self.paths["cache"])
        self.files, self.checksets = {}, {}
        self.current = None
        self.key = key                      # the Datalab key (`datalab.Key`), never in the state
        self.ledger = spend.Ledger(self.paths["runs_dir"])
        self.last_datalab = None            # time.monotonic() at the end of the last Datalab attempt
        self.server = server                # (endpoint, keys) of the marker-server route, never in the state

    @property
    def attempts(self):
        return self.state["attempts"]

    def save(self):
        save_state(self.dir, self.state)

    def resolve_documents(self):
        """Files and checks of every selected document, offline from the cache (U6: unavailable)."""
        from . import personal
        from .checks import load_or_generate
        from .fetch import resolve_document
        from .readers import read_twin
        for doc in self.docs:
            before = self.state["documents"].get(doc["id"])
            registered = before if before and before.get("status") == "available" else {}
            try:
                if self.manifest.personal:
                    # a copy is reused only with the SHA-256 the state registered (B-COPIA)
                    files = personal.resolve(doc, self.dir, registered.get("input_sha256"),
                                             registered.get("twin_sha256"))
                    body = personal.checkset(doc, files, self.dir) if doc["twin"] is not None else None
                else:
                    files = resolve_document(doc, self.manifest, self.cache, None, offline=True)
                    body = None
                    if doc["twin"] is not None:
                        body = load_or_generate(doc, files, self.cache, lambda: read_twin(doc, files, self.cache),
                                                excl_mod.feedback_for(self.exclusions, doc["id"]))
            except BenchError as exc:
                if exc.fatal:
                    raise
                self.state["documents"][doc["id"]] = {"status": "unavailable", "code": exc.code.value}
                print(f"{doc['id']} unavailable {exc.code.value}", flush=True)
                continue
            record = {"status": "available", "input_sha256": sha256_file(files.input),
                      "checks_sha256": body["checks_sha256"] if body else None,
                      "warnings": [w.value for w in files.warnings]}
            if self.manifest.personal:
                record["path"] = doc["input"]["path"]          # stays in the state, never in the report
                if files.hashes.get("twin") is not None:
                    record["twin_sha256"] = files.hashes["twin"]
            if before and before.get("status") == "available" and (
                    (before["input_sha256"], before["checks_sha256"]) != (record["input_sha256"], record["checks_sha256"])
                    or ("twin_sha256" in before and before["twin_sha256"] != record.get("twin_sha256"))):
                raise BenchError(Code.RESUME_IDENTITY_CHANGED, doc["id"] + ": input, twin or checks")
            self.state["documents"][doc["id"]] = record
            self.files[doc["id"]], self.checksets[doc["id"]] = files, body
        self.save()

    def load_checksets(self):
        """`rebuild_report`: the check set of each available document with a twin, the one whose SHA-256
        the state recorded, from the cache of the run (`RUN/checks/` for a personal set); never generated
        again. A missing, ambiguous or different one is `report_checks_changed`."""
        index = None
        for doc in self.docs:
            ds = self.state["documents"].get(doc["id"]) or {}
            if ds.get("status") != "available" or doc["twin"] is None:
                continue
            want = ds.get("checks_sha256")
            if self.manifest.personal:
                body = read_checkset(self.dir / "checks" / f"{doc['id']}.json")
            else:
                index = checks_index(self.cache.checks_dir) if index is None else index
                body = index.get((doc["id"], want))
            if want is None or body is None or body["checks_sha256"] != want or body.get("document") != doc["id"]:
                raise BenchError(Code.REPORT_CHECKS_CHANGED, doc["id"])
            self.checksets[doc["id"]] = body

    def verify_results(self):
        """`rebuild_report`: the result of the final generation of each key is the one the state
        recorded (`result_verified`); returns how many, or raises `report_result_changed`."""
        verified = 0
        for ids in self.state["keys"].values():
            a = self.attempts[ids[-1]]
            if a["status"] not in ("done", "stopped"):
                continue                     # unresolved: no result; started or abandoned: not in a finished run
            if not result_verified(a, self.dir / "attempts" / a["id"] / "out", self.state["identity"]):
                raise BenchError(Code.REPORT_RESULT_CHANGED, a["id"])
            verified += 1
        return verified

    # --- the ledger ---------------------------------------------------------------------------

    def ledger_entries(self):
        if self.readonly:
            return spend.read_only(self.ledger.path)
        with self.ledger.locked(create=False) as ledger:
            return ledger.entries() if ledger is not None else []

    def complete(self, a, result):
        """Append the completion of the reservation of a Datalab attempt (Δ6), once; the code that
        stops the run when the ledger cannot be read or written, else None."""
        result = result if isinstance(result, dict) else {}
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        transit = result.get("transit") if isinstance(result.get("transit"), dict) else {}
        try:
            entry = {"kind": "complete", "run_id": self.state["run_id"], "attempt_id": a["id"],
                     "billed_cents": spend.completion_billed(result), "status": result.get("status"),
                     "code": error.get("code"), "request_id": transit.get("request_id"),
                     "deleted": transit.get("deleted"), "at": spend.now()}
            with self.ledger.locked() as ledger:
                if not spend.has(ledger.entries(), "complete", self.state["run_id"], a["id"]):
                    ledger.append(entry)
        except BenchError as exc:
            return exc.code
        except OSError:
            return Code.SPEND_LEDGER_UNWRITABLE
        return None

    def settle(self, a, out, result, reserved):
        """The end of a Datalab attempt that has a result, the same after the call and after an adoption
        (B-SPESA): the completion of its reservation in the ledger first, then the attempt is done,
        then its billed cost against its estimate. Returns the code that stops the run before any
        other send (the ledger, `datalab_billed_over_estimate`), else None. When the completion
        cannot be written the attempt stays as it was: a resumption adopts it and completes it.
        `datalab_billed_over_estimate` is also written in the state here, with `done`: the save that
        publishes the attempt carries it, so a death before the run stops never loses it."""
        if reserved:
            failure = self.complete(a, result)
            if failure is not None:
                return failure
        a.update(status="done", result=result_record(out))
        try:
            billed = spend.completion_billed(result)
        except BenchError as exc:
            return exc.code
        if billed is not None and Decimal(billed) > Decimal(str(a["estimate_cents"])):
            self.state["stop"] = Code.DATALAB_BILLED_OVER_ESTIMATE.value
            return Code.DATALAB_BILLED_OVER_ESTIMATE
        return None

    @staticmethod
    def announce(a, what):
        """The line of an attempt that a resumption closes (R4.1): `adopted` with its status and code,
        or `unresolved datalab_unresolved`; closed codes only."""
        label = a["route"] + (":" + a["variant"]["datalab_mode"] if a["route"] == DATALAB else
                              ":" + a["variant"]["server_mode"] if a["route"] == MARKER_SERVER else "")
        if what == "adopted":
            r = a.get("result") or {}
            detail = r.get("status") if r.get("status") in ("complete", "failed") else "unrecognized"
            if r.get("code") is not None:
                detail += " " + (r["code"] if r["code"] in {c.value for c in Code} else "unrecognized")
        else:
            detail = Code.DATALAB_UNRESOLVED.value
        print(f"{a['document']} {label} {a['v8']} k{a['k']} g{a['g']} {what} {detail}", flush=True)

    def datalab_cap(self, estimate):
        """`cap_usd` of the engine for one attempt: the smallest of what the spend cap of the ledger and
        the cost cap of the run leave and of the estimate (the reservation always covers what the
        engine may send), in USD."""
        caps = self.state["datalab"]
        try:
            total, run_total = spend.totals(self.ledger_entries(), self.state["run_id"])
        except BenchError as exc:
            raise Stop(exc.code) from None
        except OSError:
            raise Stop(Code.SPEND_LEDGER_UNWRITABLE) from None
        left = min(Decimal(caps["spend_cap_usd"]) * 100 - total, Decimal(caps["max_cost_usd"]) * 100 - run_total,
                   estimate)
        return format(max(left, Decimal(0)) / 100, "f")

    # --- resumption ---------------------------------------------------------------------------

    def reconcile(self, identity):
        """Adopt or abandon the generations a previous execution left started, interrupted or stopped;
        a Datalab generation whose reservation is in the ledger is never sent again: without an
        adoptable result it is `datalab_unresolved`. An adopted Datalab result is settled like the
        result of a call (`settle`). Returns the code that stops the run before any new send, or None."""
        gens = self.state["generations"]
        entries = stop = None
        for ids in self.state["keys"].values():
            a = self.attempts[ids[-1]]
            if a["status"] not in ("started", "interrupted", "stopped"):
                continue
            out = self.dir / "attempts" / a["id"] / "out"
            if a["route"] == DATALAB and entries is None:
                entries = self.ledger_entries()
            reserved = a["route"] == DATALAB and spend.has(entries, "reserve", self.state["run_id"], a["id"])
            if adoptable(a, out, identity):
                code = None
                if a["route"] == DATALAB:
                    code = self.settle(a, out, read_result(out), reserved)
                    entries = None
                else:
                    a.update(status="done", result=result_record(out))
                    if a["route"] == MARKER_SERVER:
                        code = self.server_settle(a, out)
                if a["status"] == "done":
                    a["adopted"] = True
                    gens["adopted"] += 1
                    self.save()              # done, with the stop of its billed cost, before its line
                    self.announce(a, "adopted")
                stop = stop or code
            elif reserved:
                a.update(status="unresolved", code=Code.DATALAB_UNRESOLVED.value)
                self.announce(a, "unresolved")
            else:
                a["status"] = "abandoned"
                gens["abandoned"] += 1
        self.save()
        return stop

    def billed_stop(self):
        """Before any send of a resumption (B-SPESA): `datalab_billed_over_estimate` when the ledger
        has a completion of this run billed over the estimate of its reservation, also when an earlier
        execution left that stop out of the state; the code of a ledger that cannot be read; else None."""
        try:
            over = spend.billed_over_estimate(self.ledger_entries(), self.state["run_id"])
        except BenchError as exc:
            return exc.code
        except OSError:
            return Code.SPEND_LEDGER_UNWRITABLE
        return Code.DATALAB_BILLED_OVER_ESTIMATE if over else None

    def retry_unaccepted(self):
        """`--retry-unaccepted`: a Datalab attempt whose POST was refused before its acceptance
        (`datalab_rejected` without a `request_id`, closed at 0 in the ledger) is converted again."""
        for ids in self.state["keys"].values():
            a = self.attempts[ids[-1]]
            if a["route"] != DATALAB or a["status"] != "done":
                continue
            r = read_result(self.dir / "attempts" / a["id"] / "out") or {}
            error = r.get("error") if isinstance(r.get("error"), dict) else {}
            transit = r.get("transit") if isinstance(r.get("transit"), dict) else {}
            if r.get("status") == "failed" and error.get("code") == Code.DATALAB_REJECTED.value \
                    and not transit.get("request_id"):
                a["status"] = "rejected"     # saved by the execution: a refused resumption keeps nothing of it

    def pending_datalab(self):
        """True when a Datalab key of the run may still be converted (and sent)."""
        for key in self.keys:
            if key["route"] != DATALAB:
                continue
            ids = self.state["keys"].get(key_name(key["document"], key["route"], key["v8"], key["k"])) or []
            if not ids or self.attempts[ids[-1]]["status"] not in TERMINAL:
                return True
        return False

    # --- conversions --------------------------------------------------------------------------

    def space(self):
        """At least `DATALAB_SPACING` seconds between two Datalab attempts."""
        if self.last_datalab is not None:
            wait = self.last_datalab + DATALAB_SPACING - time.monotonic()
            if wait > 0:
                time.sleep(wait)

    def convert_all(self):
        run8 = self.state["run_id"][:8]
        for key in self.keys:
            doc_id = key["document"]
            if self.state["documents"].get(doc_id, {}).get("status") != "available":
                continue
            name = key_name(doc_id, key["route"], key["v8"], key["k"])
            ids = self.state["keys"].setdefault(name, [])
            if ids and self.attempts[ids[-1]]["status"] in TERMINAL:
                continue
            g = len(ids) + 1
            aid = attempt_id(run8, doc_id, key["route"], key["v8"], key["k"], g)
            doc = self.manifest.document(doc_id)
            remote = key["route"] == DATALAB
            if remote:
                self.space()                 # before the attempt exists; then the cap, right before the call
                estimate = datalab_estimate(doc, key["variant"]["datalab_mode"])
                request = {"route": DATALAB, "attempt": aid,
                           "datalab": {"mode": key["variant"]["datalab_mode"], "transit_confirmed": True,
                                       "cap_usd": self.datalab_cap(estimate)}}
            elif key["route"] == MARKER_SERVER:
                request = {"route": MARKER_SERVER, "attempt": aid, "ocr": key["variant"]["ocr"],
                           "server": {"name": self.state["server"]["name"], "mode": key["variant"]["server_mode"],
                                      "transit_confirmed": True}}
            else:
                request = {"route": key["route"], "attempt": aid, **key["variant"]}
            hook, extra = None, {}
            if key["route"] == MARKER_SERVER:
                # the budget and the session go into the request before the state saves it (MFL033 G-B2):
                # the state holds what engine.json will record, so a death during the conversion adopts it
                hook = self.server_hook(request, aid)
                extra = {"server_endpoint": self.server[0], "server_keys": self.server[1]}
            self.attempts[aid] = {"id": aid, "document": doc_id, "route": key["route"], "variant": key["variant"],
                                  "v8": key["v8"], "k": key["k"], "g": g, "status": "started",
                                  "input_sha256": self.state["documents"][doc_id]["input_sha256"],
                                  "request": request, "caps": self.state["caps"][doc_id], "adopted": False,
                                  "result": None, "started_at": now()}
            if remote:
                self.attempts[aid]["estimate_cents"] = str(estimate)
            ids.append(aid)
            self.save()
            out = self.dir / "attempts" / aid / "out"
            out.parent.mkdir(parents=True, mode=0o700)
            observer = DatalabSend(self, self.attempts[aid], doc["pages"], estimate) if remote else hook
            self.current = aid
            try:
                result = engine_adapter.convert(self.files[doc_id].input, out, request, self.state["caps"][doc_id],
                                                self.dir / "work", observer=observer, name=input_name(doc),
                                                model_root=str(self.paths["model_root"]),
                                                datalab_key=self.key if remote else None, **extra)
            finally:
                if remote:
                    self.last_datalab = time.monotonic()
            self.current = None
            code = (result.get("error") or {}).get("code") if isinstance(result.get("error"), dict) else None
            a, stop = self.attempts[aid], None
            if key["route"] == MARKER_SERVER:
                if observer is not None:
                    observer.complete(read_result(out) or result)
                if code in SERVER_STOPS:
                    # the session or a cap stopped the attempt before the server woke: converted again later
                    a.update(status="stopped", result=result_record(out))
                    stop = Code(code)
                else:
                    a.update(status="done", result=result_record(out))
            elif not remote:
                a.update(status="done", result=result_record(out))
            elif observer.failure is not None or code == Code.DATALAB_CAP.value:
                # a cap or the ledger stopped the send before the POST: nothing billed, converted again later
                failure = self.complete(a, read_result(out) or result) if observer.reserved else None
                a.update(status="stopped", result=result_record(out))
                stop = failure or observer.failure or Code.DATALAB_CAP
            else:
                stop = self.settle(a, out, read_result(out) or result, observer.reserved)
            a["ended_at"] = now()
            self.save()
            seconds = (result.get("timings") or {}).get("total_seconds")
            route = key["route"] + (":" + key["variant"]["datalab_mode"] if remote else
                                    ":" + key["variant"]["server_mode"] if key["route"] == MARKER_SERVER else "")
            print(f"{doc_id} {route} {key['v8']} k{key['k']} g{g} {result.get('status')}"
                  + (f" {code}" if code else "") + (f" {seconds} s" if seconds is not None else ""), flush=True)
            if stop is not None:
                raise Stop(stop)

    def server_settle(self, a, out):
        """The completion of an adopted attempt on a rented server, when its reservation has none."""
        from . import serverspend
        if self.server is None or not self.server[0].costs:
            return None
        try:
            with self.ledger.locked() as ledger:
                serverspend.complete(ledger, self.server[0].provider, self.state["run_id"], a["id"], read_result(out))
        except BenchError as exc:
            return exc.code
        except OSError:
            return Code.SPEND_LEDGER_UNWRITABLE
        return None

    def server_hook(self, request, aid):
        """For an endpoint with a provider: the session of the command (`MDF_SESSION_ARM`), the
        budget of the attempt (what the cap of the run and the window leave) in the request, and the
        reservation hook; None otherwise. A session that is not there stops the run."""
        from ..engine import servers
        from . import serverspend
        ep = self.server[0]
        if not ep.costs:
            return None
        info = self.state["server"]
        try:
            entries = self.ledger_entries()
            session = serverspend.session(self.ledger.path, ep.provider, servers.canonical_url(ep.url), entries=entries,
                                          rate=ep.rate_usd_per_s)
            spent = serverspend.run_total(entries, ep.provider, self.state["run_id"])
        except BenchError as exc:
            raise Stop(exc.code) from None
        except OSError:
            raise Stop(Code.SPEND_LEDGER_UNWRITABLE) from None
        left = max(Decimal(0), Decimal(info["max_cost_usd"]) * 100 - spent)
        budget = min(left, Decimal(session["window_left_cents"]))
        request["server"].update(budget_cents=str(budget),
                                 session={"arm_id": session["arm_id"], "armed_until": session["armed_until"]})
        return serverspend.Reservation(self.ledger, ep.provider, session["arm_id"], self.state["run_id"], aid, left)

    def interrupted(self):
        if self.current is not None:
            self.attempts[self.current]["status"] = "interrupted"
        self.state["status"] = "interrupted"


def _check_out(out, runs_dir):
    out, runs_dir = Path(out).absolute(), Path(runs_dir).absolute()
    if out.resolve() == runs_dir.resolve() or not out.resolve().is_relative_to(runs_dir.resolve()):
        raise BenchError(Code.INVALID_SETTING, "--out must be a directory under the runs directory")
    return out


def datalab_options(m, routes, settings, modes=None, confirm_transit=False, max_cost_usd=None, spend_cap_usd=None,
                    key_file=None, retry_unaccepted=False):
    """The Datalab options of a new run, checked before any file is opened (Δ5, C-ORDINE): None
    without `--route datalab` (then every Datalab option, the key file included, is refused and
    the file is never opened); the personal set is refused first."""
    if routes is None or DATALAB not in routes:
        if modes is not None or confirm_transit or max_cost_usd is not None or spend_cap_usd is not None \
                or key_file is not None or retry_unaccepted:
            raise BenchError(Code.INVALID_SETTING, "Datalab options without --route datalab")
        return None
    if m.personal:
        raise BenchError(Code.DATALAB_REFUSED_PERSONAL, "a personal set is never sent to Datalab")
    if retry_unaccepted:
        raise BenchError(Code.INVALID_SETTING, "--retry-unaccepted resumes a run (--resume)")
    modes = parse_modes(modes)
    if not confirm_transit:
        raise BenchError(Code.TRANSIT_NOT_CONFIRMED, "the documents would leave the machine for Datalab")
    if max_cost_usd is None:
        raise BenchError(Code.INVALID_SETTING, "--max-cost-usd with --route datalab")
    cap = spend_cap_usd if spend_cap_usd is not None else settings.cap_datalab_usd
    if cap is None:
        raise BenchError(Code.INVALID_SETTING, "--spend-cap-usd (or MDF_BENCH_CAP_DATALAB_USD) with --route datalab")
    return {"modes": modes, "max_cost_usd": str(parse_usd(max_cost_usd, "--max-cost-usd")),
            "spend_cap_usd": str(parse_usd(cap, "--spend-cap-usd"))}


def server_options(m, routes, modes=None, server=None, servers_file=None, confirm_transit=False, max_cost_usd=None,
                   spend_cap_usd=None, key_file=None):
    """The marker-server options of a new run (MFL033): None without `--route marker-server` (its
    options are then refused). One endpoint per run and no Datalab beside it; a personal set never
    goes outside the loopback (`server_refused_personal`, before the keys); a server outside the
    loopback needs `--confirm-transit`; an endpoint with a provider `--max-cost-usd` and
    `--spend-cap-usd`. Returns (options for the state, (endpoint, keys))."""
    from ..engine import servers
    if routes is None or MARKER_SERVER not in routes:
        if modes is not None or server is not None or servers_file is not None:
            raise BenchError(Code.INVALID_SETTING, "server options without --route marker-server")
        return None, None
    if DATALAB in routes:
        raise BenchError(Code.INVALID_SETTING, "marker-server and datalab in one run")
    modes = parse_server_modes(modes)
    path = Path(servers_file).absolute() if servers_file is not None else servers.default_path()
    from ..engine import containers
    ep = servers.load(path, server, presets=gpu_presets(), host_loopback=not containers.in_product_image())
    if m.personal and ep.leaves_machine():
        raise BenchError(Code.SERVER_REFUSED_PERSONAL, "a personal set never leaves the loopback")
    if ep.leaves_machine() and not confirm_transit:
        raise BenchError(Code.TRANSIT_NOT_CONFIRMED, "the page images would leave the machine for " + ep.host())
    caps = {}
    if ep.costs:
        if max_cost_usd is None or spend_cap_usd is None:
            raise BenchError(Code.SERVER_LEDGER_REQUIRED, "--max-cost-usd and --spend-cap-usd with " + ep.provider)
        caps = {"max_cost_usd": str(parse_usd(max_cost_usd, "--max-cost-usd")),
                "spend_cap_usd": str(parse_usd(spend_cap_usd, "--spend-cap-usd"))}
    elif max_cost_usd is not None or spend_cap_usd is not None:
        raise BenchError(Code.INVALID_SETTING, "spending caps for an endpoint without a provider")
    keys = servers.read_keys(ep, key_file, environ=None)
    return ({"name": ep.name, "servers_file": str(path), "identity_sha256": ep.identity_sha256(),
             "record": ep.record(), "modes": modes, **caps}, (ep, keys))


def gpu_presets():
    """The table of the GPU presets that an endpoint of servers.json may name (MFL023 A3), as the engine reads it."""
    from ..gpu import presets
    return presets.endpoint_table()


def load_server(state, key_file):
    """The endpoint of a run on resumption, from its file and name, with the same identity, and its keys. As at
    the start of a run (`server_options`), an endpoint on the loopback of the host is refused from a container of
    the product (CARB-NB2): a run started on the host with a `local` endpoint is not resumed in one."""
    from ..engine import containers, servers
    info = state["server"]
    ep = servers.load(info["servers_file"], info["name"], presets=gpu_presets(),
                      host_loopback=not containers.in_product_image())
    if ep.identity_sha256() != info["identity_sha256"]:
        raise BenchError(Code.RESUME_IDENTITY_CHANGED, "the endpoint " + info["name"])
    return ep, servers.read_keys(ep, key_file, environ=None)


def read_datalab_key(key_file):
    """The Datalab key, from the key file only (never the environment, never a setting; Δ5)."""
    from ..engine.datalab import read_key
    return read_key(key_file, environ={})


def check_ledger(runs_dir):
    """A ledger that does not reconcile is refused before anything is sent (or a run created)."""
    with spend.Ledger(runs_dir).locked(create=False) as ledger:
        spend.reconcile(ledger.entries() if ledger is not None else [])


def start(settings, *, level=None, docs=None, routes=None, ocr=None, repeat=None, out=None, manifest=None,
          exclusions=None, datalab_modes=None, confirm_transit=False, max_cost_usd=None, spend_cap_usd=None,
          key_file=None, retry_unaccepted=False, venv=AUTO, server_modes=None, server=None, servers_file=None):
    """A new run: validations and refusals first (manifest and selection, the personal set on Datalab,
    transit and caps, the ledger, the key), then the directory, its lock and the first state."""
    venv = venv_root() if venv is AUTO else venv
    venv_fd = lock_venv(venv)
    try:
        check_venv_users(venv)
        m = _manifest(manifest)
        ids = [x.strip() for x in docs.split(",") if x.strip()] if docs else None
        if (level is None) == (ids is None) or (m.personal and level is not None):
            raise BenchError(Code.INVALID_SETTING, "--level or --docs (--docs for a personal set)")
        selected = m.select(level=level, ids=ids)
        wanted = parse_routes(routes)
        srv, endpoint = server_options(m, wanted, server_modes, server, servers_file, confirm_transit, max_cost_usd,
                                       spend_cap_usd, key_file)
        if srv is None:
            dl = datalab_options(m, wanted, settings, datalab_modes, confirm_transit, max_cost_usd, spend_cap_usd,
                                 key_file, retry_unaccepted)
        else:
            dl = datalab_options(m, wanted, settings, datalab_modes, retry_unaccepted=retry_unaccepted)
        excl_data, excl_sha = _exclusions(exclusions)
        ocr = ocr or "auto"
        repeat = 1 if repeat is None else repeat
        if not 1 <= repeat <= MAX_REPEAT:
            raise BenchError(Code.INVALID_SETTING, "--repeat")
        keys = plan_keys(selected, wanted, ocr, repeat, dl["modes"] if dl else None, srv["modes"] if srv else None)
        run_id = secrets.token_hex(8)
        runs_dir = Path(settings.runs_dir)
        run_dir = _check_out(out if out is not None else runs_dir / run_id, runs_dir)
        from ..settings import Settings
        model_root = str(Path(Settings().model_root).absolute())
        identity = engine_adapter.identity(model_root)
        key = None
        if dl is not None or (srv is not None and endpoint[0].costs):
            check_ledger(runs_dir)
        if dl is not None:
            key = read_datalab_key(key_file)
        runs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        run_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            run_dir.mkdir(mode=0o700)
        except FileExistsError:
            raise BenchError(Code.RUN_EXISTS, "the run directory exists") from None
        lock_fd = lock_new(run_dir)
        try:
            run_dir = run_dir.absolute()
            manifest_copy = copy_input(run_dir, manifest, "manifest.json", m.sha256)
            exclusions_copy = copy_input(run_dir, exclusions, "exclusions.json", excl_sha)
            cache_dir = Path(settings.cache_dir).absolute()
            state = {"schema": STATE_SCHEMA, "run_id": run_id, "created_at": now(), "status": "running",
                     "selection": {"level": level, "docs": [d["id"] for d in selected], "routes": wanted, "ocr": ocr,
                                   "repeat": repeat, "set": m.data["set"],
                                   "datalab_modes": dl["modes"] if dl else None,
                                   "server_modes": srv["modes"] if srv else None},
                     "manifest": {"path": manifest_copy, "sha256": m.sha256},
                     "exclusions": {"path": exclusions_copy, "sha256": excl_sha},
                     "cache": str(cache_dir), "runs_dir": os.path.relpath(runs_dir.absolute(), run_dir),
                     "origin": {"run_dir": str(run_dir), "runs_dir": str(runs_dir.absolute())},
                     "model_root": model_root, "caps": {d["id"]: caps_of(d) for d in selected},
                     "datalab": None if dl is None else {k: dl[k] for k in ("max_cost_usd", "spend_cap_usd")},
                     "server": srv,
                     "identity": identity, "environment": environment(identity), "documents": {}, "keys": {},
                     "attempts": {}, "generations": {"adopted": 0, "abandoned": 0}, "executions": []}
            save_state(run_dir, state)
            paths = {"manifest": manifest, "exclusions": exclusions, "cache": cache_dir, "model_root": Path(model_root),
                     "runs_dir": runs_dir.absolute(), "in_place": True}
            return _execute(Run(run_dir, state, m, excl_data, selected, keys, key, paths=paths, server=endpoint),
                            resumed=False)
        finally:
            os.close(lock_fd)
    finally:
        if venv_fd is not None:
            os.close(venv_fd)


def resume(run_dir, confirm_transit=False, key_file=None, retry_unaccepted=False, venv=AUTO, cache=None,
           model_root=None):
    """Resume a run: its lock before its state, every option from the state (Δ7). The transit and the
    key are asked again (they are never kept) when a Datalab key remains to be converted. `cache`
    and `model_root` replace the hints of the state (MFL044 (d)), verified by what it recorded."""
    venv = venv_root() if venv is AUTO else venv
    venv_fd = lock_venv(venv)
    try:
        check_venv_users(venv)
        run_dir = Path(run_dir).absolute()
        lock_fd = lock_existing(run_dir)
        try:
            state = load_state(run_dir)
            sel = state["selection"]
            paths = run_paths(run_dir, state, cache=cache, model_root=model_root)
            need(paths, "cache")
            m = _manifest(paths["manifest"])
            remote = DATALAB in (sel["routes"] or [])
            served = MARKER_SERVER in (sel["routes"] or [])
            if served and not isinstance(state.get("server"), dict):
                raise BenchError(Code.RESUME_STATE_INVALID, "the endpoint of the run")
            rented = served and state["server"].get("max_cost_usd") is not None
            if rented and not paths["in_place"]:
                raise BenchError(Code.RUN_PATH_MISSING, "a run on a rented server is resumed where it was created, by its ledger")
            if remote and not paths["in_place"]:
                raise BenchError(Code.RUN_PATH_MISSING, "a run with Datalab is resumed where it was created, by its ledger")
            if remote and m.personal:
                raise BenchError(Code.DATALAB_REFUSED_PERSONAL, "a personal set is never sent to Datalab")
            if not remote and (retry_unaccepted or (not served and (confirm_transit or key_file is not None))):
                raise BenchError(Code.INVALID_SETTING, "Datalab options for a run without Datalab")
            if remote and not (isinstance(state.get("datalab"), dict) and sel.get("datalab_modes")):
                raise BenchError(Code.RESUME_STATE_INVALID, "Datalab caps or modes")
            excl_data, excl_sha = _exclusions(paths["exclusions"])
            changed = []
            if m.sha256 != state["manifest"]["sha256"]:
                changed.append("manifest")
            if excl_sha != state["exclusions"]["sha256"]:
                changed.append("exclusions")
            if not changed:
                selected = m.select(ids=sel["docs"])
                keys = selection_keys(selected, sel)
                if any(k["route"] in WEIGHTED_ROUTES for k in keys):
                    need(paths, "model_root")        # before the identity, which would only see the weights missing
            identity = engine_adapter.identity(str(paths["model_root"]))
            changed = _identity_differences(state["identity"] or {}, identity) + changed
            if changed:
                raise BenchError(Code.RESUME_IDENTITY_CHANGED, ", ".join(changed))
            endpoint = None
            if served:
                endpoint = load_server(state, key_file)
                if m.personal and endpoint[0].leaves_machine():
                    raise BenchError(Code.SERVER_REFUSED_PERSONAL, "a personal set never leaves the loopback")
                if endpoint[0].leaves_machine() and not confirm_transit:
                    raise BenchError(Code.TRANSIT_NOT_CONFIRMED, "the page images would leave the machine")
            run = Run(run_dir, state, m, excl_data, selected, keys, paths=paths, server=endpoint)
            stop = run.reconcile(identity)
            if stop is None and state.get("stop") in STICKY_STOPS:
                stop = Code(state["stop"])           # the maintainer has not decided yet: nothing is sent
            if stop is None and remote:
                stop = run.billed_stop()             # the ledger too: a stop no state recorded stays
            if stop is not None:
                return _execute(run, resumed=True, stop=stop)
            if retry_unaccepted:
                run.retry_unaccepted()
            if run.pending_datalab():
                if not confirm_transit:
                    raise BenchError(Code.TRANSIT_NOT_CONFIRMED, "the documents would leave the machine for Datalab")
                check_ledger(paths["runs_dir"])
                run.key = read_datalab_key(key_file)
            state["status"] = "running"
            state["stop"] = None
            return _execute(run, resumed=True)
        finally:
            os.close(lock_fd)
    finally:
        if venv_fd is not None:
            os.close(venv_fd)


def _execute(run, resumed, stop=None):
    """One execution of a run; with `stop` (a stop a resumption found before any send) it resolves the
    documents, then stops with that code, the state and the report saved."""
    from . import report as report_mod
    execution = {"started_at": now(), "resumed": resumed}
    run.state["executions"].append(execution)
    run.save()
    if run.key is not None:
        print(DATALAB_WARNING, flush=True)
    try:
        run.resolve_documents()
        if stop is not None:
            raise Stop(stop)
        run.convert_all()
        # the wheel is verified again at the end: a report of a changed package is not valid (§8)
        identity_changed = engine_adapter.package_identity() != run.state["identity"]["package"]
        unavailable = sum(1 for d in run.state["documents"].values() if d["status"] == "unavailable")
        code = 1 if identity_changed else 2 if unavailable else 0
        execution.update(ended_at=now(), exit=code, peak_rss_bytes=_peak_rss(), identity_changed=identity_changed)
        run.state["status"] = "complete"
        report_mod.write(run.dir, report_mod.build(run, identity_changed=identity_changed))
    except KeyboardInterrupt:
        run.interrupted()
        execution.update(ended_at=now(), exit=1, peak_rss_bytes=_peak_rss())
        run.save()
        raise BenchError(Code.INTERRUPTED, "interrupted") from None
    except Stop as stopped:
        # a cap, the ledger or a billed cost over its estimate: nothing more is sent; state and report kept
        run.state["status"] = "stopped"
        run.state["stop"] = stopped.code.value
        execution.update(ended_at=now(), exit=1, peak_rss_bytes=_peak_rss(), stop=stopped.code.value)
        run.save()
        try:
            report_mod.write(run.dir, report_mod.build(run))
        except (OSError, BenchError):
            pass
        raise BenchError(stopped.code, "the run stopped") from None
    run.save()
    print(f"run {run.state['run_id']}: {len(run.state['keys'])} keys, {unavailable} documents unavailable"
          + (", identity changed: report not valid" if identity_changed else ""), flush=True)
    if identity_changed:
        raise BenchError(Code.IDENTITY_CHANGED, "the package changed during the run")
    return code


def rebuild_report(run_dir, venv=AUTO, report_dir=None, cache=None, manifest=None, exclusions=None):
    """`mdf bench report RUN_DIR`: `report.json` and `report.md` of a finished run written again from
    its saved outputs, with the scoring of this package; nothing is converted or sent (no engine, no
    Datalab, no key) and neither the state nor the outputs are written. Under the lock of the run
    (`run_busy`), and only when everything still matches, each refusal a closed code: the run ended
    (`complete` or `stopped`; otherwise `report_run_not_finished`); its manifest and exclusions have
    the SHA-256 of its state (`report_identity_changed`); the check set of each document is the one
    whose SHA-256 the state recorded, read and never generated (`report_checks_changed`); the result
    of each final generation is the `engine.json` the state recorded, of the same attempt, input and
    options, with components that agree with the identity of the run and, when complete, every file
    of its inventory (`report_result_changed`), as an adoption checks it.

    With `report_dir` (MFL044 (d)) the run is only read: no lock and no write in it; `report_dir`
    is new or empty and outside the run, and receives `report.json`, `report.md` and the outcome
    of each check (`attempts/<id>/results.json`); the state is read again before the report is
    written and must have the same SHA-256 (`report_run_changed`, nothing written). `cache`,
    `manifest` and `exclusions` replace the paths of the state, verified by its SHA-256."""
    from . import report as report_mod
    venv = venv_root() if venv is AUTO else venv
    venv_fd = lock_venv(venv)
    try:
        run_dir = Path(run_dir).absolute()
        target = None
        if report_dir is not None:
            target = Path(report_dir).absolute()
            if target.resolve().is_relative_to(run_dir.resolve()):
                raise BenchError(Code.INVALID_SETTING, "--report-dir must be outside the run")
            if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
                raise BenchError(Code.INVALID_SETTING, "--report-dir must be a new or empty directory")
        lock_fd = lock_existing(run_dir) if target is None else None
        try:
            before = _state_sha256(run_dir) if target is not None else None
            state = load_state(run_dir)
            if state.get("status") not in ("complete", "stopped"):
                raise BenchError(Code.REPORT_RUN_NOT_FINISHED, "the run did not end: resume it")
            sel = state["selection"]
            paths = run_paths(run_dir, state, cache=cache, manifest=manifest, exclusions=exclusions)
            try:
                m = _manifest(paths["manifest"])
                excl_data, excl_sha = _exclusions(paths["exclusions"])
            except BenchError:
                raise BenchError(Code.REPORT_IDENTITY_CHANGED, "manifest or exclusions unreadable") from None
            if m.sha256 != state["manifest"]["sha256"] or excl_sha != state["exclusions"]["sha256"]:
                raise BenchError(Code.REPORT_IDENTITY_CHANGED, "manifest or exclusions")
            selected = m.select(ids=sel["docs"])
            keys = selection_keys(selected, sel)
            run = Run(run_dir, state, m, excl_data, selected, keys, paths=paths, readonly=target is not None)
            if any(state["documents"].get(d["id"], {}).get("status") == "available" and d["twin"] is not None
                   for d in selected) and not m.personal:
                need(paths, "cache")
            run.load_checksets()
            verified = run.verify_results()
            last = state["executions"][-1] if state["executions"] else {}
            # an execution of an earlier version does not record it: complete with exit 1 means a changed package
            identity_changed = last.get("identity_changed", state["status"] == "complete" and last.get("exit") == 1)
            rebuilt = {"at": now(), "package": engine_adapter.package_identity()}
            if target is None:
                report_mod.write(run_dir, report_mod.build(run, identity_changed=identity_changed, rebuilt=rebuilt))
            else:
                created = not target.exists()
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    built = report_mod.build(run, identity_changed=identity_changed, rebuilt=rebuilt, out_dir=target)
                    if _state_sha256(run_dir) != before:
                        raise BenchError(Code.REPORT_RUN_CHANGED, "the state of the run changed during the report")
                    report_mod.write(target, built)
                except BaseException:
                    import shutil
                    shutil.rmtree(target / "attempts", ignore_errors=True)
                    if created:
                        try:
                            target.rmdir()
                        except OSError:
                            pass
                    raise
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
    finally:
        if venv_fd is not None:
            os.close(venv_fd)
    where = "report.json and report.md written again" if target is None else "report written to the report directory"
    print(f"report {state['run_id']}: {verified} results verified, {where} (nothing converted)", flush=True)
    return 0


def _state_sha256(run_dir):
    try:
        fd = os.open(Path(run_dir) / STATE, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        raise BenchError(Code.RESUME_STATE_MISSING, "no state of a run") from None
    with os.fdopen(fd, "rb") as fh:
        return hashlib.sha256(fh.read(MAX_STATE_BYTES + 1)).hexdigest()


def _peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
