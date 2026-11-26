"""Durable single-host worker with explicit retries and publication fencing.

Every file of the archive goes through the conversion engine (`engine.convert`) on the route of
its job: the default of the conversion matrix or the one the user chose, with the options
validated again by `services.job_options`, the function of the API (MFL032 C3, MFL023 A5). A route that cannot run on this
machine fails its job with `route_unavailable` and the queue goes on. The Datalab route runs only
with the transit confirmed in the job, under the cap `MDF_DATALAB_CAP_USD` of each conversion,
with the key `DATALAB_API_KEY` read by name from the keys file at every use (MFL051; never from the
environment, out of which the worker still takes that name when it starts
(never a setting, never passed to a child, never written).

A job on the route marker-server (MFL023 B4) runs on the GPU session of its preset (`gpu.lifecycle`): the
worker opens it at the first job of the preset (or at a request to arm ahead), serves the jobs of that preset
first, lingers, then closes it; every call to a provider runs in a child (`gpu.opchild`); a window left by a dead
worker is closed before the first claim. The reservation of each attempt is chained to the observer, its completion
written in every outcome, and the facts of the attempt on its session go into `runs.gpu` before the publication.

Every child of the engine is a container started by the launcher (MFL051, `engine/containers.py`).
The observer of the engine renews the lease, records the running child (`runs.child`: the name of
its container) and passes on cancellation and stop. A container is named to the observer before
it is created, and the attempt keeps the monotonic indicator `runs.scope_recorded` (0 until then,
1 after).

One rule releases the space commitment of an attempt (B1): its record in `operations/` goes away
only after the launcher attested that no container of the attempt is left (`stop_attempt_children`:
list by the prefix of its names, kill, remove, confirmed absence). It holds at the end of a job
(`commitment`), in the loop (`release_kept`: while a commitment is kept the worker claims no job,
says so in its heartbeat and asks again every few seconds) and at the start (`_reconcile`: the
prefixes to attest come from the records without a live owner, whatever the state of their job,
and from the rows still running with the indicator raised). A launcher or a daemon that does not
answer is no proof that a child is gone: the commitment is kept and the error is visible.

A child the launcher itself stopped (the service going down or restarting), or one whose caller
stopped renewing its lease, ends its conversion with `interrupted`: the job becomes `interrupted`,
terminal and open to `mdf retry`, as when the worker is stopped. The full result of the engine
stays next to the published result (`engine-result.json`); the manifest carries a bounded summary
of it (K9).
"""
from pathlib import Path
from contextlib import contextmanager
import json
import os
import shutil
import sqlite3
import logging
import time
import uuid

from decimal import Decimal

from . import routestate
from .engine import containers, isolation, routes, selfprobe
from .gpu import codes as gpu_codes, lifecycle
from .engine.codes import Code, EngineError
from .engine.core import availability, convert
from .engine.formats import extension_of
from .engine.progress import finishing
from .files import identify, make_directory, open_file, read_json, rename, sync_directory, write_json, sync_tree, corruption, relative, IntegrityError, MetadataError, decode_json, publication_barrier, directory, FileSafetyError, json_bytes
from .references import inspect
from .services import Refused, Service, ReadUnavailable, job_options, source_format, stored_manifest, stored_options, publication_budget
from .storage import StorageError, CapacityError

MAX_ISSUES = 200                 # engine problems listed in the manifest; all of them are counted
MANIFEST_LIMIT = 1048576         # the JSON limit of files.decode_json
READY_SECONDS = 60               # how long a check of the availability of the converters holds
ENGINE_FILES = ("engine-out", "engine-work")
KEPT_SECONDS = 5.0               # how often the worker asks again for an attestation it did not get


class Unattested(StorageError):
    """Docker did not attest, through the launcher, that no child of an attempt is left: its commitment is kept."""


class Interrupted(StorageError):
    """A child of the engine was stopped from outside the worker (the stop of the launcher, a lease not renewed):
    the job is `interrupted`, not `failed`."""


def execute(settings, scratch, module, inputs=(), tick=None, deadline=60, memory_mib=512, output_mib=None,
            label="process", attempt=None, args=(), results=None):
    """A light child of the engine (`engine/isolation.py`) for the raster validation and the isolation probe: a
    container without network, `inputs` read-only, its own scratch the only writable directory. `scratch/label`
    is a directory of the worker: what is in it goes to the child, and `results` come back into it. Returns a
    receipt: `exit`, `reason` (a closed code of the engine or None), `asked` (the tick of the worker asked for the
    stop), `elapsed_seconds`, `dir` (where the results of the child are)."""
    work = Path(scratch) / label
    started = time.monotonic()
    asked = []

    def observer(event):
        if tick is None:
            return None
        stop = tick({"container": event["container"]} if event.get("container") else None, None)
        stop = "interrupted" if stop == "interrupted" else "cancelled" if stop else None
        if stop:
            asked.append(stop)
        return stop
    try:
        rc, _ = isolation.run(work, module, args, inputs=inputs, memory_mib=memory_mib,
                              output_mib=output_mib or settings.output_mib, timeout=deadline,
                              observer=observer, poll_seconds=0.2, attempt=attempt, results=results)
        reason = None
    except EngineError as exc:
        rc, reason = None, exc.code.value
    return {"exit": rc, "reason": reason, "asked": bool(asked), "elapsed_seconds": time.monotonic() - started, "dir": work}


def stop_attempt_children(attempt_id):
    """Attest that no child of an attempt is left, by the prefix of the names of its containers
    (`mdf-eng-<attempt[:16]>-`): list, kill, remove, confirmed absence; then the folders of its calls go away.
    `Unattested` when the launcher or Docker cannot say so: no proof that a child is gone, the commitment of the
    attempt is kept (K7, HC1, PB8 of MFL032; B1 of MFL051). Also for an attempt whose worker died before it
    recorded a child."""
    prefix = containers.prefix_of(attempt_id)
    try:
        launcher = containers.launcher()
        containers.stop_attempt_children(launcher, prefix)
    except (containers.Unattested, containers.Refused) as exc:
        raise Unattested('Children of the attempt not attested absent (%s*: %s); reservation kept' % (prefix, exc)) from None
    containers.sweep(launcher, containers.WORK_ROOT, prefix=prefix, min_age=0)


def chain(*observers):
    """The observers of one conversion in order: the first stop wins (as `cli.Chain`)."""
    def observe(event):
        for observer in observers:
            if observer is not None:
                stop = observer(event)
                if stop:
                    return stop
        return None
    return observe


def gpu_message(code):
    """The error of a run refused or stopped by its GPU session (a closed code)."""
    text = 'Conversion failed: %s; source kept, retry available' % code
    if code == gpu_codes.MODAL_NO_CAPACITY:
        text += '; ' + lifecycle.NO_CAPACITY_TEXT
    return text


class Observer:
    """The observer of one conversion (K7): every event of the engine may renew the lease; the
    child record changes before the child does anything. An exception of the lease or of the
    database is kept and stops the engine (`interrupted`), then raised again by the worker."""

    def __init__(self, worker, row):
        self.worker, self.row = worker, row
        self.child = self.progress = self.error = None
        self.last = 0.0
        self.interval = max(0.2, worker.settings.poll_seconds / 2)

    def __call__(self, event):
        if self.error is not None:
            return "interrupted"
        kind, changed = event.get("event"), False
        if kind == "child" or (kind == "tool" and event.get("container")):
            # named before the container is created; an event of a child is never throttled
            self.child, changed = ({"container": event["container"]} if event.get("container") else {}), True
        elif kind == "progress" and isinstance(event.get("progress"), dict):
            self.progress = event["progress"]
        now = time.monotonic()
        if not changed and now - self.last < self.interval:
            return None
        self.last = now
        try:
            stop = self.worker.tick(self.row, self.child, self.progress)
        except Exception as exc:            # raised again after the engine has stopped its child
            self.error = exc
            return "interrupted"
        return "interrupted" if stop == "interrupted" else "cancelled" if stop else None


def take_gpu_keys(settings, environ=None):
    """The keys of the rented GPUs taken out of the environment of the worker (MFL023, C-NB2 of the
    plan r1): the names of `gpu.providers.KEYS` and the `key_env` of the endpoints of the instance's
    servers.json, so that no child inherits them; the worker reads its keys only from its env file.
    A file that cannot be read, or entries of another type, are skipped (C-DA-NB3 of the review of A); of
    `key_env` only the names of keys leave, by the rule of `engine.servers`: ending in one of
    `envfile.KEY_SUFFIXES`, never an `MDF_` setting. Returns the names taken, never a value."""
    from .engine.envfile import KEY_SUFFIXES
    from .gpu.providers import KEYS
    environ = os.environ if environ is None else environ
    names = set(KEYS)
    try:
        doc = json.loads(Path(settings.servers).read_bytes())
        entries = doc.get("servers") if isinstance(doc, dict) else None
    except (OSError, ValueError):
        entries = None
    for entry in (entries.values() if isinstance(entries, dict) else ()):
        given = entry.get("key_env") if isinstance(entry, dict) else None
        given = [given] if isinstance(given, str) else given if isinstance(given, list) else []
        names.update(n for n in given if isinstance(n, str) and not n.startswith("MDF_") and n.endswith(KEY_SUFFIXES))
    return sorted(n for n in names if environ.pop(n, None) is not None)


def take_datalab_key(environ=None):
    """`DATALAB_API_KEY` taken out of the environment of the process, so no child can inherit it. The value is
    dropped (MFL051): the key of the worker is read by name from its keys file at every use, never from the
    environment. Returns whether the name was there, never the value."""
    from .engine import datalab
    return (os.environ if environ is None else environ).pop(datalab.KEY_NAME, None) is not None


def engine_message(code, settings):
    """The error of a run for a closed code of the engine."""
    if code == Code.STALLED.value:
        return ('No conversion progress for %d s (MDF_STALL_SECONDS); attempt stopped, '
                'source kept, retry available' % settings.stall_seconds)
    if code == Code.TIMEOUT.value:
        return ('Conversion exceeded %d s (MDF_TIMEOUT_SECONDS); attempt stopped, '
                'source kept, retry available' % settings.timeout_seconds)
    return 'Conversion failed: %s; source kept, diagnostics in the attempt, retry available' % code


def source_name(name, source_path):
    """The name the engine records for the source: the user's name when its extension is the one
    of the archived source, else the name of the archived file."""
    ok = isinstance(name, str) and '/' not in name and extension_of(name) == extension_of(source_path)
    return name if ok else None


def engine_request(options, attempt, fmt=None, cap_usd="0", settings=None):
    """The request of the engine for the stored options of a run (`services.stored_options`: a
    run of schema 2 maps `force_ocr` to `ocr: force`; `strip_existing_ocr` has no equivalent and
    is dropped, D3 of MFL032: an existing OCR layer is set aside by the engine itself). With `fmt`
    the options are validated again by `services.job_options`, the function of the API, with the
    settings of the instance (EngineError or `Refused` when refused); the Datalab route gets the
    confirmed transit of the job and the cap of the settings; the GPU route (MFL023 B3) its server
    mode and the confirmed transit, to which the session of its GPU preset adds the endpoint, the
    budget and the window."""
    opts = stored_options(options)
    if fmt is not None:
        opts = job_options(fmt, opts, settings)
    request = {"route": opts.get("route"), "ocr": opts.get("ocr", "auto"), "lang": opts.get("lang"), "attempt": attempt}
    converter = routes.resolve(fmt, opts["route"])[0] if fmt is not None else None
    if converter == routes.DATALAB:
        request["datalab"] = {"mode": opts["datalab_mode"], "transit_confirmed": opts["transit_confirmed"],
                              "cap_usd": cap_usd}
    elif converter == routes.MARKER_SERVER:
        request["server"] = {"mode": opts["server_mode"], "transit_confirmed": opts["transit_confirmed"]}
    return request


def summarize(result, inventory):
    """The bounded summary of an engine result that the manifest carries (K9, PB9): every count,
    at most MAX_ISSUES problems (the incomplete ones first, as the engine lists them), the page
    classes without the per-page records, the corrections counted only, the timings without the
    seconds of each page (their count, median and maximum stay)."""
    summary = {k: result[k] for k in ("schema", "status", "attempt", "started_at", "source", "options", "limits",
                                      "problem_codes", "problem_counts", "components", "timings", "transit", "cost")
               if k in result}
    timings = summary.get("timings")
    if isinstance(timings, dict) and isinstance(timings.get("pages"), dict):
        # the statistics of the seconds per page, not the value of each page
        summary["timings"] = dict(timings, pages={k: v for k, v in timings["pages"].items() if k != "values"})
    route = result.get("route") or {}
    summary["route"] = {k: route.get(k) for k in ("converter", "family", "chosen_by", "reason")}
    if isinstance(result.get("pages"), dict):
        summary["pages"] = {k: result["pages"].get(k) for k in ("count", "classes")}
    summary["corrections"] = (result.get("corrections") or {}).get("counts", {})
    problems = result.get("problems") or []
    summary["problems"] = problems[:MAX_ISSUES]
    summary["problems_listed"] = len(summary["problems"])
    summary["inventory"] = inventory
    summary["inventory_complete"] = True
    return summary


def engine_issues(engine):
    """The problems of an engine summary as `checks.issues` (K9): `code` incomplete or unchecked,
    a readable `reason` that names the page (the Web UI reads `reason`), `path` and `page`.
    The severities come from the counts of all the problems, before any truncation (PB9)."""
    issues = []
    for p in engine.get("problems") or []:
        if not isinstance(p, dict) or not isinstance(p.get("code"), str):
            raise StorageError("Malformed engine problems")
        reason = p["code"].replace("_", " ")
        if isinstance(p.get("page"), int):
            reason += " on page %d" % p["page"]
        if isinstance(p.get("path"), str):
            reason += ": " + p["path"]
        if isinstance(p.get("detail"), str):
            reason += " (" + p["detail"] + ")"
        issue = {"code": "incomplete" if p.get("severity") == "incomplete" else "unchecked", "reason": reason[:500]}
        for key in ("path", "page"):
            if p.get(key) is not None:
                issue[key] = p[key]
        issues.append(issue)
    counts = engine.get("problem_counts") or {}
    total = sum(v for v in counts.values() if type(v) is int)
    if total > len(issues):
        issues.append({"code": "incomplete" if counts.get("incomplete") else "unchecked",
                       "reason": "%d more problems of the engine, counted in engine-result.json" % (total - len(issues))})
    return issues, counts


class Worker:
    def __init__(self, settings, env_file=None, gpu=None):
        self.settings = settings
        # the env file of `mdf worker --env-file`, checked by the CLI before Settings.load: the keys
        # of the GPU sessions are read from it by name, never from the environment (MFL023)
        # (MFL051: the keys file, `--keys-file`). A file that is not safe does not stop the worker: every read
        # refuses it, and the routes with a key are unavailable with their closed code
        self.env_file = None if env_file is None else str(Path(env_file).absolute())
        self.service = Service(settings)
        self.storage = self.service.storage
        self.token = uuid.uuid4().hex
        self.stop = False
        self.lease = None
        self._worker_locked = False
        self._ready_at = None
        self.kept = {}                   # attempt: the commitment kept until the absence of its children is attested
        self._kept_at = None
        self.converters = None           # availability per converter, from `ready`
        self.readiness = {"ready": False, "message": "Worker starting", "converters": None}
        take_datalab_key()
        take_gpu_keys(settings)
        self.launcher_instance = None    # the launcher the probe ran against (`hello`: `instance`)
        self.probes = 0
        self._code = None                # the code this process runs (`containers.identity`), read once
        self.key_details = {}            # {converter: what to do about its key}, never a value
        self._refresh_seen = self.service.routes_refresh_asked()      # a request older than this worker is its start
        # the GPU sessions (MFL023 B4): `gpu` gives the tests the command of the children, the clocks and the follower
        self.gpu = lifecycle.Manager(self, **(gpu or {}))
        self._gpu_job = None

    def datalab_key(self):
        """The key of Datalab, read now from the keys file (`datalab_key_unsafe`, `datalab_key_missing`)."""
        from .engine import datalab
        return datalab.read_key(self.env_file)

    def datalab_state(self):
        """Datalab can run here with its key in the keys file, read now by name, and a cap above 0."""
        from .engine import datalab
        state = datalab.key_state(self.env_file)
        self.key_details.pop(routes.DATALAB, None)
        if not state["available"]:
            if state["code"] == Code.DATALAB_KEY_UNSAFE.value:
                self.key_details[routes.DATALAB] = state["detail"]
            return {"available": False, "code": state["code"]}
        if Decimal(self.settings.datalab_cap_usd) <= 0:
            return {"available": False, "code": Code.DATALAB_CAP.value}
        return {"available": True, "code": None}

    def ready(self):
        """The availability of every converter on this machine (`engine.availability`: the
        launcher, the toolchain image and the verified weights of Marker;
        Datalab from the keys file and the cap), checked again after READY_SECONDS; the
        engine verifies the weights again before every Marker child. Returns it, or None when the children would
        not run the code of this worker: the launcher answers with another build of the package (`stale engine
        image`), or it is another launcher than the one of the probe and the probe, repeated, fails. The worker
        then takes no job (`turn`) and says why in its heartbeat."""
        now = time.monotonic()
        if self._ready_at is not None and now - self._ready_at < READY_SECONDS:
            return self.converters
        failed = self.same_engine()
        if failed:
            self.converters, self._ready_at = None, now
            message = "Isolation unavailable; no fallback (%s)" % ", ".join(failed)[:200]
            self.readiness = {"ready": False, "code": Code.ISOLATION_UNAVAILABLE.value, "converters": None,
                              "message": message, "routes": self.routes_blocked(message)}
            return None
        state = availability(model_root=self.settings.model_root, keys_file=self.env_file)
        state[routes.DATALAB] = self.datalab_state()
        missing = sorted('%s (%s)' % (name, s['code']) for name, s in state.items() if not s['available'])
        notes = ''.join('; %s: %s' % item for item in sorted(self.key_details.items()))
        self.converters, self._ready_at = state, now
        self.readiness = {"ready": True, "converters": state,
                          "message": "Local worker ready" + ("; routes unavailable: " + ", ".join(missing) if missing else "")
                                     + notes}
        if self.settings.gpu_presets_enabled():
            self.readiness["gpu"] = self.gpu.readiness()      # closed values: availability, minimum reserve, ledger
        self.readiness["routes"] = self.routes_now(state)
        return state

    def gpu_providers(self):
        """{GPU preset the instance enables: its provider}."""
        names = self.settings.gpu_presets_enabled()
        if not names:
            return {}
        from .gpu import presets as gpu_presets
        table = gpu_presets.load()
        return {name: table[name]["provider"] for name in names if name in table}

    def routes_now(self, state):
        """The state of every route (`routestate`, MFL051) from what this worker just checked: the converters, the
        keys file (names only), the GPU presets and their ledgers. It goes in the heartbeat; the API reads it."""
        gpu = self.readiness.get("gpu") or {}
        presets = gpu.get("presets") or {}
        details = dict(self.key_details)
        unsafe = self.key_details.get(routes.DATALAB)             # the same file holds the keys of the GPUs
        for name, item in presets.items():
            if item.get("code") == Code.SERVER_KEY_UNSAFE.value and unsafe:
                details["%s:%s" % (routestate.MARKER_SERVER, name)] = unsafe
        key_names = {name: item["missing_keys"] for name, item in presets.items() if item.get("missing_keys")}
        return routestate.compute(state, self.settings.datalab_cap_usd, gpu, self.gpu_providers(), details, key_names,
                                  now=time.time())

    def routes_blocked(self, detail):
        """The routes of a worker that takes no job: every one broken, with the reason of the worker."""
        return routestate.blocked(Code.ISOLATION_UNAVAILABLE.value, self.settings.datalab_cap_usd,
                                  self.settings.gpu_presets_enabled(), self.gpu_providers(), detail, now=time.time())

    @contextmanager
    def exclusive(self):
        if self._worker_locked:
            yield
            return
        with self.storage.lock('worker.lock', blocking=False):
            self._worker_locked = True
            try:
                yield
            finally:
                self._worker_locked = False

    def heartbeat(self, readiness=None):
        """Record the worker alive with its readiness (a JSON object: `ready`, `message`, the
        availability of each converter, read by /api/v1/readiness and /api/v1/session)."""
        value = json.dumps(readiness or self.readiness, allow_nan=False)
        with self.storage.connect() as conn:
            conn.execute("INSERT INTO worker VALUES (1,?,?,?) ON CONFLICT(id) DO UPDATE SET token=excluded.token,heartbeat=excluded.heartbeat,readiness=excluded.readiness", (self.token, time.time(), value))

    def expire_heartbeat(self):
        """At a clean exit (MFL023 B10, SOL-B-N2 and C-DB-NB2.1 of the review of the diff B): the heartbeat of this
        worker expires, so the API no longer shows its readiness (GPU presets, caps) for the lease; only while the row
        is still its own: a worker that ends never touches the heartbeat of the one after it."""
        with self.storage.connect() as conn:
            conn.execute("UPDATE worker SET heartbeat=0 WHERE id=1 AND token=?", (self.token,))

    def reconcile(self, run_id=None):
        with self.exclusive():
            self._reconcile(run_id)

    def _reconcile(self, run_id):
        from .capacity import recover, live_workers, dead_workers
        rows = [self.storage.get(r['id']) for r in self.storage.list()
                if run_id is None or r['id'] == run_id]
        with self.storage.lock('admission.lock'):
            live = live_workers(self.storage)
            dead = dead_workers(self.storage)
        selected, attest = [], set()
        # Attest the absence of the children of the dead commitments before releasing them.
        # Live commitments from independent operations remain locked and reserved.
        for row in rows:
            if row['attempt_id'] in live and row['token'] != self.token:
                continue
            selected.append(row)
            if row['state'] in {'running', 'publishing'} and row.get('scope_recorded') != 0:
                # 1: the attempt named a container before creating it; NULL: an attempt of a previous version,
                # without the indicator. 0: no child was ever started, nothing to ask
                attest.add(row['attempt_id'])
        # B1: the commitments without a live owner, WHATEVER the state of their job (also `failed` or `completed`,
        # also without a row: the probe of a worker that died), are released only after the attestation
        attest |= dead if run_id is None else dead & {row['attempt_id'] for row in selected}
        for attempt in sorted(attest):
            stop_attempt_children(attempt)
        with self.storage.lock('admission.lock'):
            recover(self.storage, workers=True if run_id is None else
                    {row['attempt_id'] for row in selected})
        for row in selected:
            if row['state'] == 'completed' and row['bundle'] in {'ready', 'unchecked'}:
                try:
                    self.service.verified_files(row['id'])
                except ReadUnavailable as exc:
                    logging.warning('Reconciliation read unavailable for %s: %s', row['id'], exc)
                except StorageError:
                    pass
            if row["state"] not in {"running", "publishing"}:
                continue
            if row["state"] == "publishing":
                try:
                    present = []
                    for location in ('attempts', 'results'):
                        try:
                            (self.storage.root / location / row['attempt_id']).lstat()
                        except FileNotFoundError:
                            continue
                        present.append(location)
                    if len(present) != 1:
                        raise IntegrityError('Publication location conflict or missing result')
                    moved = present[0] == 'results'
                    result = self.storage.root / present[0] / row['attempt_id']
                    with directory(result):
                        pass
                    try:
                        manifest = stored_manifest(row['manifest'], row)
                        if read_json(result, 'manifest.json') != manifest:
                            raise IntegrityError('Publication manifest differs from database')
                    except MetadataError as exc:
                        raise IntegrityError('Malformed publication manifest') from exc
                    if not row['token']:
                        raise IntegrityError('Publication ownership revoked')
                    with directory(result / 'original'):
                        pass
                    actual = set()
                    def failed(exc):
                        raise exc
                    for parent, dirs, files in os.walk(result / 'original', followlinks=False, onerror=failed):
                        if any((Path(parent) / name).is_symlink() for name in dirs):
                            raise IntegrityError('Unsafe publication directory')
                        actual.update((Path(parent) / name).relative_to(result / 'original').as_posix()
                                      for name in files)
                    if actual != {item['path'] for item in manifest['artifacts']}:
                        raise IntegrityError('Publication inventory mismatch')
                    for item in manifest['artifacts']:
                        observed = identify(result / 'original', item['path'])
                        if any(observed[k] != item[k] for k in ('bytes', 'sha256')):
                            raise IntegrityError('Publication file mismatch')
                    source = self.storage.get(row['id'])
                    actual_source = identify(self.storage.root, source['source_path'])
                    if actual_source['sha256'] != source['source_sha256'] or actual_source['bytes'] != source['source_bytes']:
                        raise IntegrityError('Publication source mismatch')
                    with self.storage.reservation('control', publication_budget(row['manifest'])):
                        self.complete_publication(row, manifest, moved=moved)
                    continue
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    if not corruption(exc):
                        self.defer_publication(row, exc)
                        continue
                    reason = 'Publication integrity failure: ' + str(exc) + '; source and partials preserved'
            else:
                reason = 'Worker interrupted; source and partials preserved'
            with self.storage.connect() as conn:
                conn.execute("UPDATE runs SET state='interrupted',token=NULL,error=?,updated=? WHERE id=? AND token IS ? AND state=?",
                             (reason, time.time(), row['id'], row['token'], row['state']))

    def claim(self):
        # with a GPU session open on a preset, its jobs first (MFL023 B4): a local conversion never keeps a GPU armed
        preset = self.gpu.claim_preset()
        with self.storage.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if preset is None:
                row = conn.execute("SELECT id FROM runs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            else:
                row = conn.execute("SELECT id FROM runs WHERE state='queued' ORDER BY "
                                   "(json_extract(options,'$.gpu_preset') = ?) DESC, created LIMIT 1", (preset,)).fetchone()
            if row is None:
                return None
            # a queued run never started an attempt: one of a previous version gets the indicator too
            conn.execute("UPDATE runs SET state='running',token=?,lease=?,updated=?,scope_recorded=COALESCE(scope_recorded,0) WHERE id=? AND state='queued'", (self.token, time.time() + self.settings.lease_seconds, time.time(), row["id"]))
        return self.storage.get(row["id"])

    def tick(self, row, child, progress=None):
        """Renew the lease in the same update that records the child and, when given,
        the progress snapshot; without one the last snapshot is kept."""
        self.heartbeat()
        with self.storage.connect() as conn:
            current = conn.execute("SELECT token,cancel FROM runs WHERE id=?", (row["id"],)).fetchone()
            if current["token"] != self.token:
                return True
            # a container is named here before it is created: the indicator of the attempt becomes 1 in the
            # same update, and never goes back (a trigger of 003.sql)
            conn.execute("UPDATE runs SET lease=?,child=?,updated=?,progress=COALESCE(?,progress),"
                         "scope_recorded=CASE WHEN ? THEN 1 ELSE scope_recorded END WHERE id=? AND token=?",
                         (time.time() + self.settings.lease_seconds, json.dumps(child, allow_nan=False) if child else None, time.time(),
                          json.dumps(progress, allow_nan=False) if progress else None,
                          bool(child and (child.get('container') or child.get('scope'))), row["id"], self.token))
        if self.stop:
            return 'interrupted'
        if self._gpu_job == row["id"] and not self.gpu.serving():
            return 'interrupted'                 # the GPU session closed under the job (MFL023 B4)
        if self.lease:
            self.lease.check(local_only=True)
        return bool(current["cancel"])

    def commitment(self, row):
        return self.committed(row['attempt_id'], (2 * self.settings.output_mib + self.settings.log_mib + 1) * 1048576,
                              ('attempts/' + row['attempt_id'], 'results/' + row['attempt_id']))

    @contextmanager
    def committed(self, attempt, budget, paths):
        """The space commitment of an attempt (a job, or the probe of the worker), released under the one rule of
        B1: when the block ends, however it ends, the launcher must attest that no child of the attempt is left
        (ordinarily a list that comes back empty). When it does not, the commitment is KEPT (`Lease.keep`): its
        record stays in `operations/`, its budget stays committed, and `release_kept` asks again; the outcome
        of the job is written as usual, whatever it is."""
        with self.storage.reservation('worker', budget, attempt, paths) as lease:
            try:
                yield lease
            finally:
                try:
                    stop_attempt_children(attempt)
                except Unattested as exc:
                    logging.warning('%s', exc)
                    self.kept[attempt] = lease.keep()

    def release_kept(self, now=None):
        """Ask again for the attestations that were missing, at most every `KEPT_SECONDS`; each commitment is
        released only after its own. True when none is kept any more: only then the worker claims a job."""
        if not self.kept:
            return True
        now = time.monotonic() if now is None else now
        if self._kept_at is not None and now - self._kept_at < KEPT_SECONDS:
            return False
        self._kept_at = now
        for attempt in sorted(self.kept):
            try:
                stop_attempt_children(attempt)
            except Unattested:
                continue
            self.kept.pop(attempt).close()
        if not self.kept:
            self._kept_at = None
        return not self.kept

    def pending(self):
        """The heartbeat of a worker that holds commitments whose children are not attested absent."""
        prefixes = sorted(containers.prefix_of(attempt) for attempt in self.kept)
        message = ("Cleanup of a previous conversion not attested by Docker (%s): no job is taken until it is; "
                   "if it lasts, run `make restart`" % ", ".join(prefixes))
        return {"ready": False, "code": Code.ISOLATION_UNAVAILABLE.value, "pending": prefixes, "converters": None,
                "message": message, "routes": self.routes_blocked(message)}

    def process(self, row):
        with self.exclusive():
            self._owned_process(row)

    def _owned_process(self, row):
        try:
            with self.commitment(row) as lease:
                self.lease = lease
                self._process(row)
        except (OSError, ValueError, KeyError, TypeError, IndexError, sqlite3.OperationalError) as exc:
            self.failure(row, exc)
        finally:
            self.lease = None

    def failure(self, row, exc):
        # Finite recovery only. A non-writable DB must remain explicitly unresolved
        # until restart; never claim that a terminal state has been persisted.
        for attempt in range(3):
            try:
                current = self.storage.get(row['id'])
                if current['state'] == 'publishing':
                    if current['token'] != row['token']:
                        raise StorageError('Publication ownership lost; archive preserved')
                    self.reconcile(row['id'])
                    return
                with self.storage.connect() as conn:
                    current = conn.execute('SELECT cancel FROM runs WHERE id=?', (row['id'],)).fetchone()
                    state = ('cancelled' if current['cancel'] else
                             'interrupted' if self.stop or isinstance(exc, Interrupted) else 'failed')
                    error = str(exc) if isinstance(exc, (StorageError, MetadataError, FileSafetyError)) else 'Conversion failed: ' + type(exc).__name__
                    conn.execute("UPDATE runs SET state=?,error=?,token=NULL,updated=? WHERE id=? AND token=? AND state='running'",
                                 (state, error, time.time(), row['id'], self.token))
                return
            except sqlite3.OperationalError as db_exc:
                if attempt == 2 or not any(word in str(db_exc).lower() for word in ('locked', 'busy')):
                    raise StorageError('Database unavailable; worker stopped; running job requires reconciliation on restart') from db_exc
                time.sleep(0.05 * (attempt + 1))

    def _process(self, row):
        scratch = self.storage.root / 'attempts' / row['attempt_id']
        make_directory(self.storage.root, 'attempts/' + row['attempt_id'])
        source = identify(self.storage.root, row['source_path'])
        if source['sha256'] != row['source_sha256'] or source['bytes'] != row['source_bytes']:
            raise StorageError('Archived source identity mismatch')
        options = json.loads(row['options'])
        fmt = source_format(row['source_path'])
        try:
            request = engine_request(options, row['attempt_id'], fmt, self.settings.datalab_cap_usd, self.settings)
        except EngineError as exc:          # validated by the API too: a job of another version
            raise StorageError('Conversion refused: %s; source kept' % exc.code.value) from None
        except Refused as exc:              # a GPU preset the instance no longer enables (MFL023 B3)
            raise StorageError('Conversion refused: %s; source kept, retry available' % exc.code) from None
        converter = routes.resolve(fmt, request['route'])[0]
        state = (self.converters or {}).get(converter)
        if state is not None and not state['available']:
            # the queue goes on: only this job fails, with the reason of the route
            raise StorageError('Conversion failed: %s (%s: %s); source kept, retry available when the route is ready'
                               % (Code.ROUTE_UNAVAILABLE.value, converter, state['code']))
        write_json(scratch, 'request.json', {'options': options, 'engine_request': request,
                   'dropped_options': ['strip_existing_ocr'] if options.get('strip_existing_ocr') else []})
        limits = self.settings.engine_limits()
        observer = Observer(self, row)
        asked = []                              # the stops this worker itself asked of the engine

        def watched(inner):
            def observe(event):
                stop = inner(event)
                if stop:
                    asked.append(stop)
                return stop
            return observe
        datalab_key = None
        if converter == routes.DATALAB:
            # the key is read from the keys file now, for this conversion: a file made unsafe, replaced or removed
            # since the last check of the routes is seen here, before anything is sent
            try:
                datalab_key = self.datalab_key()
            except EngineError as exc:
                raise StorageError('Conversion failed: %s (%s: %s); source kept, retry available when the route is ready'
                                   % (Code.ROUTE_UNAVAILABLE.value, converter, exc.code.value)) from None
        extra, hook, gpu = {}, None, converter == routes.MARKER_SERVER
        if gpu:
            opts = job_options(fmt, stored_options(options), self.settings)
            try:
                ep, keys, srv, hook = self.gpu.prepare(row, opts, request)
            except lifecycle.Refusal as exc:
                self.gpu.refused(row, opts['gpu_preset'], exc)
                raise StorageError(gpu_message(exc.code)) from None
            request = dict(request, server=srv)
            extra = {'server_endpoint': ep, 'server_keys': keys}
            self._gpu_job = row['id']
        result = None
        try:
            result = convert(self.storage.root / row['source_path'], scratch / 'engine-out', request, limits,
                             work=scratch / 'engine-work', observer=watched(chain(hook, observer) if gpu else observer),
                             name=source_name(row.get('name'), row['source_path']),
                             model_root=self.settings.model_root, datalab_key=datalab_key, **extra)
        finally:
            if gpu:
                self._gpu_job = None
                closed = self.gpu.session is not None and not self.gpu.serving()
                reason = self.gpu.session.state.snapshot()[1] if self.gpu.session is not None else None
                self.gpu.done(row, hook, result)
        if observer.error is not None:
            raise observer.error
        if result.get('status') != 'complete':
            code = (result.get('error') or {}).get('code', 'engine_failed')
            if gpu and closed and reason in lifecycle.JOB_CODES and code in ('interrupted', 'cancelled'):
                code = lifecycle.JOB_CODES[reason]
                if code not in ('interrupted', 'cancelled'):
                    raise StorageError(gpu_message(code))
            if code == Code.INTERRUPTED.value and not asked:
                # nobody of this worker asked: the launcher stopped the child, or its lease was not renewed (A2-c)
                raise Interrupted('Conversion interrupted: a child of the engine was stopped; source and partials '
                                  'preserved, retry available')
            raise StorageError(engine_message(code, self.settings))
        self.tick(row, None, finishing(observer.progress))
        # inside the private attempt, synchronized with it before the publication (sync_tree)
        out = scratch / 'engine-out'
        if (out / 'diagnostics').is_dir():
            (out / 'diagnostics').rename(scratch / 'diagnostics')
        (out / 'engine.json').rename(scratch / 'engine-result.json')
        out.rename(scratch / 'original')
        shutil.rmtree(scratch / 'engine-work', ignore_errors=True)
        inventory = [i for i in result['inventory'] if not i['path'].startswith('diagnostics/')]
        write_json(scratch, 'engine.json', summarize(result, inventory))
        runtime = {'engine': 'markdown_for_llms.engine', 'converter': result['route']['converter'],
                   'components': result.get('components')}
        self.publish(row, scratch, runtime, {'elapsed_seconds': result['timings']['total_seconds']})

    def publish(self, row, scratch, runtime, receipt):
        with self.exclusive():
            return self._owned_publish(row, scratch, runtime, receipt)

    def _owned_publish(self, row, scratch, runtime, receipt):
        if self.lease is not None:
            return self._publish(row, scratch, runtime, receipt)
        with self.commitment(row) as lease:
            self.lease = lease
            try:
                return self._publish(row, scratch, runtime, receipt)
            finally:
                self.lease = None

    def _publish(self, row, scratch, runtime, receipt):
        engine = read_json(scratch, 'engine.json')
        validate_engine(engine)
        original = scratch / "original"
        inventory = engine["inventory"]
        if not engine["inventory_complete"] or not inventory:
            raise StorageError("Incomplete adapter inventory")
        names = [i["path"] for i in inventory]
        if len(names) != len(set(names)) or "document.md" not in names:
            raise StorageError("Invalid adapter inventory")
        paths = list(original.rglob('*'))
        if any(p.is_symlink() for p in paths):
            raise StorageError('Symlink in adapter output')
        actual = {str(p.relative_to(original)) for p in paths if not p.is_dir()}
        if actual != set(names):
            raise StorageError("Adapter inventory differs from output")
        for item in inventory:
            if identify(original, item["path"]) != item:
                raise StorageError("Adapter output identity mismatch")
        with open_file(original, "document.md") as stream:
            raw = stream.read().decode("utf-8", "strict")
        if not raw.strip():
            raise StorageError("Empty conversion output")
        checks = inspect(raw, names)
        issues, counts = engine_issues(engine)
        checks['issues'].extend(issues)
        if engine.get('issues') or counts.get('incomplete'):      # `issues`: results of the engine before MFL032
            if engine.get('issues'):
                checks['issues'].append({'code': 'incomplete', 'reason': 'Engine reported conversion errors'})
            if checks['bundle'] != 'invalid':
                checks['bundle'] = 'incomplete'
        elif counts.get('warning') and checks['bundle'] == 'ready':
            checks['bundle'] = 'unchecked'
        artifacts = [dict(item, id=uuid.uuid4().hex, gallery=False) for item in inventory]
        candidates = [item for item in inventory if item['path'] != 'document.md']
        if candidates:
            shutil.rmtree(scratch / 'raster', ignore_errors=True)
            make_directory(scratch, 'raster')
            write_json(scratch / 'raster', 'raster-request.json', dict(artifacts=candidates,
                       max_pixels=self.settings.max_pixels, max_dimension=self.settings.max_dimension))
            raster = execute(self.settings, scratch, 'markdown_for_llms.engines.raster', inputs=[(original, 'original')],
                             tick=lambda child, progress: self.tick(row, child, progress), deadline=60, label='raster',
                             attempt=row['attempt_id'], results=('raster-result.json',))
            if raster['exit'] == 0 and not raster['reason']:
                results = validate_raster(read_json(raster.get('dir', scratch), 'raster-result.json'), candidates, self.settings)
                for item in artifacts:
                    item.update(results.get(item['path'], {}))
            else:
                if raster['reason'] == 'interrupted' and not raster.get('asked'):
                    raise Interrupted('Conversion interrupted: a child of the engine was stopped; source and '
                                      'partials preserved, retry available')
                if raster['reason'] in {'interrupted', 'cancelled'}:
                    raise StorageError(raster['reason'])
                for item in artifacts:
                    item['reason'] = 'Gallery validation failed; original retained'
            for item in inventory:
                if identify(original, item['path']) != item:
                    raise StorageError('Raster validation changed original bytes')
            shutil.rmtree(scratch / 'raster', ignore_errors=True)
        manifest = dict(schema=1, run_id=row["id"], attempt_id=row["attempt_id"], document_id=row["document_id"],
                        source={"sha256": row["source_sha256"], "bytes": row["source_bytes"]},
                        state="completed", bundle=checks["bundle"], quality="unverified", artifacts=artifacts,
                        checks=checks, engine=engine, runtime=runtime, config=self.settings.public(),
                        elapsed_seconds=receipt["elapsed_seconds"],
                        limits=["Experimental conversion; formulas, OCR, reading order and semantic asset completeness require manual verification"])
        body = json_bytes(manifest)
        if len(body) > MANIFEST_LIMIT:
            raise StorageError('Manifest over %d bytes; engine-result.json kept in the attempt' % MANIFEST_LIMIT)
        # The exact bytes written to file, DB and new ZIPs share the reader contract.
        stored_manifest(body, row)
        db_body = body.decode('utf-8')
        stored_manifest(db_body, row)
        write_json(scratch, "manifest.json", manifest)
        sync_tree(scratch)
        self.lease.check()
        if self.stop:
            raise StorageError('Worker interrupted before publication')
        with self.storage.reservation('control', publication_budget(db_body)):
            with self.storage.connect() as conn:
                changed = conn.execute("UPDATE runs SET state='publishing',manifest=?,error=NULL,updated=? WHERE id=? AND token=? AND state='running' AND cancel=0", (db_body, time.time(), row["id"], self.token)).rowcount
                if changed != 1:
                    raise StorageError("Publication token revoked or cancellation requested")
            destination = self.storage.root / "results" / row["attempt_id"]
            rename(self.storage.root, 'attempts/' + row['attempt_id'], 'results/' + row['attempt_id'])
            self.complete_publication(row, manifest)

    def complete_publication(self, row, manifest, *, moved=True):
        """Caller holds a control reservation; all success paths share this barrier."""
        current = self.storage.get(row['id'])
        if current['state'] != 'publishing' or not row['token'] or current['token'] != row['token']:
            raise IntegrityError('Publication ownership lost')
        publication_barrier(self.storage.root, row['attempt_id'], moved=moved)
        self.storage.capacity()
        with self.storage.connect() as conn:
            changed = conn.execute("UPDATE runs SET state='completed',bundle=?,result_path=?,error=NULL,token=NULL,updated=? WHERE id=? AND token=? AND state='publishing'",
                                   (manifest['bundle'], 'results/' + row['attempt_id'], time.time(), row['id'], row['token'])).rowcount
            if changed != 1:
                raise StorageError('Publication ownership lost')

    def defer_publication(self, row, exc):
        reason = str(exc) if isinstance(exc, CapacityError) else 'Publication I/O or synchronization temporarily unavailable'
        logging.warning('Publication pending for %s: %s; restart worker to reconcile', row['id'], reason)
        try:
            # A failed diagnostic reservation must not grow a full archive.
            with self.storage.reservation('control', 65536), self.storage.connect() as conn:
                conn.execute("UPDATE runs SET error=?,updated=? WHERE id=? AND token=? AND state='publishing'",
                             (reason, time.time(), row['id'], row['token']))
        except (CapacityError, OSError):
            # Without room for a durable diagnosis, release worker ownership on
            # exit so passive readers can observe that recovery is deferred.
            self.stop = True

    def run(self, once=False):
        with self.exclusive():
            try:
                # MFL023 B4: the windows left by a dead worker are closed before anything else, the first claim included
                self.gpu.recover()
            except lifecycle.Unrecoverable as exc:
                self.heartbeat({"ready": False, "code": gpu_codes.GPU_UNRECOVERABLE, "commands": exc.commands,
                                "message": "A GPU window left by a previous worker cannot be closed: " + exc.detail,
                                "converters": None})
                raise StorageError(gpu_codes.GPU_UNRECOVERABLE + ": " + exc.detail) from None
            try:
                self.reconcile()
            except Unattested as exc:
                # the commitments stay: the service restarts and asks again, until the launcher answers
                self.heartbeat({"ready": False, "code": Code.ISOLATION_UNAVAILABLE.value, "message": str(exc)[:300],
                                "converters": None})
                raise
            if self.stop:
                self.expire_heartbeat()
                return
            failed = self.probe()
            if failed:
                self.heartbeat({"ready": False, "code": Code.ISOLATION_UNAVAILABLE.value, "converters": None,
                                "message": "Isolation unavailable; no fallback (%s)" % ", ".join(failed)[:200]})
                raise StorageError('Isolation readiness failed (%s); diagnostic retained' % ", ".join(failed)[:200])
            try:
                while not self.stop:
                    self.turn(once)
                    if once:
                        break
                    time.sleep(self.settings.poll_seconds)
            finally:
                self.gpu.shutdown()          # a session open: CLOSING to a final state, then the exit
                for lease in self.kept.values():
                    lease.abandon()          # the records stay: the next worker attests from them before its first claim
            # a clean exit (--once, SIGINT, SIGTERM): the heartbeat expires; a worker stopped by an error keeps it,
            # with its message (a GPU window it cannot close, the isolation)
            self.expire_heartbeat()

    def turn(self, once=False):
        """One turn of the loop: the commitments still kept first, then the readiness, the GPU session and at most
        one job. Returns the row it processed, or None."""
        if not self.release_kept():
            # B1: no claim while a commitment is kept. The worker stays alive (with Docker down nobody would
            # restart it) and the GPU session, if one is open, still closes at its term
            self.heartbeat(self.pending())
            self.gpu.poll()
            return None
        try:
            asked = self.service.routes_refresh_asked()
            if asked > self._refresh_seen:
                # `POST /api/v1/routes/refresh`, `mdf routes --refresh`: the routes are checked now, not at the next minute
                self._refresh_seen, self._ready_at = asked, None
            state = self.ready()
            self.heartbeat()
        except (ValueError, OSError, ImportError) as exc:
            self.heartbeat({"ready": False, "message": "Conversion engine unavailable: " + str(exc)[:120],
                            "converters": None})
            if once:
                raise StorageError("Conversion engine unavailable: " + str(exc)[:120]) from None
            return None
        self.gpu.poll()
        if state is None:
            # the children would not run the code of this worker (`ready`): no job is taken until they do
            if once:
                raise StorageError(self.readiness["message"])
            return None
        row = self.claim()
        if row:
            self.process(row)
        return row

    def code(self):
        """What this process runs: its code, the build inputs its image recorded, its environment. Read once: the
        modules of a process are those of its start, and the files of an image do not change."""
        if self._code is None:
            self._code = containers.identity()
        return self._code

    def launcher_code(self):
        """The code of the launcher against the code of this worker, without a container: (`hello`, what failed).
        The code always; the build inputs when this process runs from an image that recorded its own. The children
        are containers of the image of the launcher: when it differs, they would run another code than this worker
        (the launcher was rebuilt and recreated while this worker went on: `make up` recreates both)."""
        try:
            hello = containers.launcher().hello()
        except (containers.Unreachable, containers.Refused) as exc:
            return None, ["launcher: %s" % exc]
        mine = self.code()
        stale = [name for name in ("payload_sha256", "inputs_sha256")
                 if hello.get(name) != mine[name] and (name == "payload_sha256" or mine[name] is not None)]
        return hello, (["stale engine image (%s): run `make up`" % ", ".join(stale)] if stale else [])

    def same_engine(self):
        """At every check of the routes (`ready`): the comparison of the code with the launcher, and the whole
        probe again when the launcher is another one than the one of the last probe. Returns what failed. A
        launcher that does not answer is not judged here: the routes that need it say so (`availability`), the
        others still run."""
        hello, failed = self.launcher_code()
        if hello is None:
            return []
        if failed:
            return failed
        if self.launcher_instance is not None and hello.get("instance") != self.launcher_instance:
            return self.probe()
        return []

    def probe(self):
        """The probe of the worker at its start, in place of the first job: a light child of the launcher reports
        what it can and cannot do (`selfprobe`, container mode) and the code it runs. Returns what failed: the
        launcher runs another build of the package than this worker (the payload gate: run `make up`), the child
        did not run, or an entry of its report. Empty when conversion work may be admitted."""
        hello, failed = self.launcher_code()
        if failed:
            return failed
        mine = self.code()
        # the first probe keeps the name of the worker; one repeated for another launcher has a directory of its own
        name = 'worker-' + self.token + ('-%d' % self.probes if self.probes else '')
        self.probes += 1
        probe = self.storage.root / 'staging' / name
        make_directory(self.storage.root, 'staging/' + name)
        with self.committed(self.token, (2 * self.settings.output_mib + 1) * 1048576, ('staging/' + name,)):
            receipt = execute(self.settings, probe, containers.PROBE_MODULE, args=['container'], deadline=60,
                              memory_mib=256, label='probe', attempt=self.token, results=('probe.json',))
        if receipt['exit'] or receipt['reason']:
            return ["probe: exit %s, %s" % (receipt['exit'], receipt['reason'])]
        try:
            report = json.loads((receipt['dir'] / 'probe.json').read_bytes())
        except (OSError, ValueError):
            return ["probe: no report"]
        failed = selfprobe.verdict(report, uid=os.getuid(), payload_sha256=mine["payload_sha256"])
        if not failed:
            self.launcher_instance = hello.get("instance")      # only a probe that passed is not repeated
        return failed


def validate_engine(engine):
    if (not isinstance(engine, dict) or engine.get('inventory_complete') is not True
            or not isinstance(engine.get('inventory'), list) or not engine['inventory']
            or len(engine['inventory']) > 4096
            or not isinstance(engine.get('issues', []), list)):
        raise StorageError('Malformed engine metadata')
    for item in engine['inventory']:
        if (not isinstance(item, dict) or set(item) != {'path', 'bytes', 'sha256'}
                or type(item['bytes']) is not int or item['bytes'] < 0
                or not isinstance(item['sha256'], str) or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise StorageError('Malformed engine inventory')
        relative(item['path'])
    if any(not isinstance(issue, dict) for issue in engine.get('issues', [])):
        raise StorageError('Malformed engine issues')


def validate_raster(results, candidates, settings):
    if not isinstance(results, list) or len(results) != len(candidates):
        raise StorageError('Malformed raster result')
    expected = {a['path'] for a in candidates}
    verified = {}
    allowed = {'path', 'gallery', 'reason', 'mime', 'width', 'height'}
    for item in results:
        if (not isinstance(item, dict) or set(item) - allowed
                or not isinstance(item.get('path'), str) or item['path'] not in expected
                or item['path'] in verified or type(item.get('gallery')) is not bool
                or not (item.get('reason') is None or isinstance(item['reason'], str))):
            raise StorageError('Malformed raster fields')
        if item['gallery']:
            if (item.get('mime') not in {'image/png', 'image/jpeg'}
                    or any(type(item.get(k)) is not int or not 1 <= item[k] <= settings.max_dimension
                           for k in ('width', 'height'))
                    or item['width'] * item['height'] > settings.max_pixels):
                raise StorageError('Unsafe raster gallery metadata')
        elif set(item) - {'path', 'gallery', 'reason'}:
            raise StorageError('Unexpected rejected raster metadata')
        verified[item['path']] = {k: v for k, v in item.items() if k != 'path'}
    return verified
