"""The Datalab spend ledger of the bench, `<runs-dir>/spend.jsonl`: its writes and its reconciliation.

The rows of the rented inference servers of MFL033 share the file and carry `provider`; this
module reads only the rows without it (`datalab_rows`), `serverspend.py` the others.

One line per event, JSON: `{"kind": "reserve", "run_id", "attempt_id", "estimate_cents", ...}`
when a document is about to be sent (the event `datalab-send` of the engine, before the POST),
then at most one `{"kind": "complete", "run_id", "attempt_id", "billed_cents", ...}`. Each attempt
(`run_id`, `attempt_id`) counts once (delta r3 Δ1, Δ6): its completion replaces the estimate when
it carries `billed_cents`; without a completion, or with `billed_cents: null` (an uncertain end),
the estimate stays. A completion closes a reservation at 0 only for a refusal of the POST before
the request was accepted (`datalab_rejected` without a `request_id`, `completion_billed`). A
completion without its reservation, or two reservations or completions of the same attempt, is
`spend_ledger_inconsistent`. Amounts are cents, read as decimals.

Writes (plan r2 §12). The ledger is shared by every run of a runs directory, and by processes
that run at the same time: `Ledger.locked` opens it without following links (a regular file of
the user, mode 0600) and holds an exclusive `flock` from the reading of the ledger, through the
check of the caps, to the appended line and its fsync. A new run never resets it.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import json
import os
from pathlib import Path
import stat

from .codes import BenchError, Code

LEDGER = "spend.jsonl"
MAX_LEDGER_BYTES = 64 * 1048576


def _cents(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, what)
    try:
        cents = Decimal(str(value))
    except InvalidOperation:
        raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, what) from None
    if not cents.is_finite() or cents < 0:
        raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, what)
    return cents


def datalab_rows(entries):
    """The Datalab rows: those without `provider` (the rows of the rented inference servers carry it
    and are read by `serverspend.py`; MFL033)."""
    return [e for e in entries if not (isinstance(e, dict) and "provider" in e)]


def reconcile(entries):
    """{(run_id, attempt_id): cents} and the total of the Datalab rows, from the parsed lines in
    their order."""
    reserved, completed = {}, {}
    for e in datalab_rows(entries):
        if not isinstance(e, dict) or not isinstance(e.get("run_id"), str) or not isinstance(e.get("attempt_id"), str):
            raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "line without run and attempt")
        key = (e["run_id"], e["attempt_id"])
        if e.get("kind") == "reserve":
            if key in reserved:
                raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "two reservations of one attempt")
            reserved[key] = _cents(e.get("estimate_cents"), "estimate_cents")
        elif e.get("kind") == "complete":
            if key not in reserved:
                raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "completion without its reservation")
            if key in completed:
                raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "two completions of one attempt")
            billed = e.get("billed_cents")
            completed[key] = None if billed is None else _cents(billed, "billed_cents")
        else:
            raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "unknown kind")
    per_attempt = {k: completed[k] if completed.get(k) is not None else v for k, v in reserved.items()}
    return per_attempt, sum(per_attempt.values(), Decimal(0))


def parse(data):
    """The parsed lines of the bytes of a ledger; a line that is not JSON (a write cut short
    included) is `spend_ledger_inconsistent`."""
    entries = []
    if data and not data.endswith(b"\n"):
        raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "last line cut short")
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "invalid line") from None
    return entries


def read(path):
    """The parsed lines of a ledger file (none when it does not exist), without its lock."""
    path = Path(path)
    if not path.exists():
        return []
    return parse(path.read_bytes())


def read_only(path):
    """The parsed lines of a ledger file opened read-only, without its lock and without following a
    link (none when it does not exist): `mdf bench report --report-dir` writes nothing (MFL044 (d))."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except FileNotFoundError:
        return []
    except OSError:
        raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "ledger cannot be opened") from None
    with os.fdopen(fd, "rb") as fh:
        info = os.fstat(fh.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_LEDGER_BYTES:
            raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "the ledger is not a regular file within the limit")
        return parse(fh.read(MAX_LEDGER_BYTES + 1))


def totals(entries, run_id):
    """(total of the ledger, total of the run `run_id`) in cents."""
    per_attempt, total = reconcile(entries)
    return total, sum((v for (r, _), v in per_attempt.items() if r == run_id), Decimal(0))


def billed_over_estimate(entries, run_id):
    """The attempts of the run `run_id` whose completion is billed over the estimate of their
    reservation (`datalab_billed_over_estimate`), in the order of the ledger; a ledger that does not
    reconcile is `spend_ledger_inconsistent`."""
    reconcile(entries)
    entries = datalab_rows(entries)
    estimates = {e["attempt_id"]: _cents(e["estimate_cents"], "estimate_cents") for e in entries
                 if e["kind"] == "reserve" and e["run_id"] == run_id}
    return [e["attempt_id"] for e in entries if e["kind"] == "complete" and e["run_id"] == run_id
            and e.get("billed_cents") is not None
            and _cents(e["billed_cents"], "billed_cents") > estimates[e["attempt_id"]]]


def has(entries, kind, run_id, attempt_id, provider=None):
    """A row of `kind` for the attempt; `provider` None: a Datalab row."""
    return any(isinstance(e, dict) and e.get("kind") == kind and e.get("run_id") == run_id
               and e.get("attempt_id") == attempt_id and e.get("provider") == provider for e in entries)


def completion_billed(result):
    """`billed_cents` of the completion of an attempt from its `engine.json` (Δ6): the billed cost
    when the result has it; 0 for a POST refused before its acceptance (`datalab_rejected` and no
    `request_id`); otherwise None, and the estimate stays (`datalab_unavailable`, `datalab_failed`,
    `interrupted`, `timeout`, `datalab_rejected` with a `request_id`, `datalab_cost_unknown`, ...)."""
    result = result if isinstance(result, dict) else {}
    cost = result.get("cost") if isinstance(result.get("cost"), dict) else {}
    billed = cost.get("billed_cents")
    if billed is not None:
        return str(_cents(billed, "billed_cents"))
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    transit = result.get("transit") if isinstance(result.get("transit"), dict) else {}
    if error.get("code") == Code.DATALAB_REJECTED.value and not transit.get("request_id"):
        return "0"
    return None


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_line(fd, data):
    """Append one line and make it durable (a function of its own, so that a test can make it fail)."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]
    os.fsync(fd)


class Locked:
    """The ledger under its exclusive lock."""

    def __init__(self, fd, path):
        self.fd, self.path = fd, path

    def entries(self):
        size = os.fstat(self.fd).st_size
        if size > MAX_LEDGER_BYTES:
            raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "ledger too large")
        data, offset = bytearray(), 0
        while offset < size:
            chunk = os.pread(self.fd, min(1 << 20, size - offset), offset)
            if not chunk:
                break
            data += chunk
            offset += len(chunk)
        return parse(bytes(data))

    def append(self, entry):
        _write_line(self.fd, (json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii"))


class Ledger:
    """`<runs-dir>/spend.jsonl`, or the file given with `Ledger.file` (`mdf convert --spend-ledger`)."""

    def __init__(self, runs_dir):
        self.path = Path(runs_dir) / LEDGER

    @classmethod
    def file(cls, path):
        ledger = cls.__new__(cls)
        ledger.path = Path(path)
        return ledger

    @contextmanager
    def locked(self, create=True):
        """The ledger under an exclusive `flock`. With `create` it is created (0600) when absent, and
        its directory made durable; without, an absent ledger yields None."""
        flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        created = False
        try:
            fd = os.open(self.path, flags)
        except FileNotFoundError:
            if not create:
                yield None
                return
            try:
                fd = os.open(self.path, flags | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
            except FileExistsError:          # another process created it in the meantime
                fd = os.open(self.path, flags)
        except OSError:
            raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "ledger cannot be opened") from None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise BenchError(Code.SPEND_LEDGER_INCONSISTENT, "the ledger is not a private regular file of the user")
            if created:
                dfd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield Locked(fd, self.path)
        finally:
            os.close(fd)
