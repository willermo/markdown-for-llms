"""Durable single-host worker with explicit retries and publication fencing.

Every file of the archive goes through the conversion engine (`engine.convert`) on the route of
its job: the default of the conversion matrix or the one the user chose, with the options
validated again by `engine.routes.job_options` (MFL032 C3). A route that cannot run on this
machine fails its job with `route_unavailable` and the queue goes on. The Datalab route runs only
with the transit confirmed in the job, under the cap `MDF_DATALAB_CAP_USD` of each conversion,
with the key `DATALAB_API_KEY` that the worker takes out of its own environment when it starts
(never a setting, never passed to a child, never written).

The observer of the engine renews the lease, records the running child (`runs.child`: the pid
with its start time, the systemd scope of the Marker child, the container of a tool) and passes
on cancellation and stop; reconciliation stops what a dead worker left by that record. A scope is
named to the observer before it starts, and the attempt keeps the monotonic indicator
`runs.scope_recorded` (0 until then, 1 after): reconciliation asks the systemd user manager only
for the attempts that recorded a scope, and for those of a previous version without the
indicator (NULL). The full result of the engine stays next to the published result
(`engine-result.json`); the manifest carries a bounded summary of it (K9).
"""
from pathlib import Path
from contextlib import contextmanager
import json
import os
import shutil
import signal
import sqlite3
import logging
import time
import uuid

from decimal import Decimal

from .engine import isolation, routes
from .engine.codes import Code, EngineError
from .engine.core import availability, convert
from .engine.formats import extension_of
from .engine.progress import finishing
from .files import identify, make_directory, open_file, read_json, rename, sync_directory, write_json, sync_tree, corruption, relative, IntegrityError, MetadataError, decode_json, publication_barrier, directory, FileSafetyError, json_bytes
from .references import inspect
from .services import Service, ReadUnavailable, source_format, stored_manifest, stored_options, publication_budget
from .storage import StorageError, CapacityError

MAX_ISSUES = 200                 # engine problems listed in the manifest; all of them are counted
MANIFEST_LIMIT = 1048576         # the JSON limit of files.decode_json
READY_SECONDS = 60               # how long a check of the availability of the converters holds
ENGINE_FILES = ("engine-out", "engine-work")


def process_identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start": raw[19], "state": raw[0]}
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def execute(settings, scratch, module, inputs=(), tick=None, deadline=60, memory_mib=512, output_mib=None,
            label="process"):
    """A light child of the engine (`engine/isolation.py`) for the raster validation and the
    isolation probe: Bubblewrap without network, read-only system and `inputs`, `scratch/label`
    its only writable directory. Returns a receipt: `exit`, `reason` (a closed code of the
    engine or None), `elapsed_seconds`, `dir` (where the child wrote its results)."""
    work = Path(scratch) / label
    started = time.monotonic()

    def observer(event):
        if tick is None:
            return None
        child = {"pid": event["pid"]} if event.get("pid") else None
        if child and (ident := process_identity(event["pid"])):
            child["start"] = ident["start"]
        stop = tick(child, None)
        return "interrupted" if stop == "interrupted" else "cancelled" if stop else None
    try:
        rc, _ = isolation.run(work, module, inputs=inputs, memory_mib=memory_mib,
                              output_mib=output_mib or settings.output_mib, timeout=deadline,
                              observer=observer, poll_seconds=0.2)
        reason = None
    except EngineError as exc:
        rc, reason = None, exc.code.value
    return {"exit": rc, "reason": reason, "elapsed_seconds": time.monotonic() - started, "dir": work}


def stop_attempt_scopes(attempt_id):
    """Stop every scope of the Marker child that an attempt may have left, by the prefix of its
    names (`mdf-eng-<attempt[:16]>-`), also when the worker died before recording one (HC1); an
    unreachable manager keeps the reservation, as for a recorded scope (PB8)."""
    prefix = 'mdf-eng-%s-' % attempt_id[:16]
    try:
        names = isolation.list_scopes(prefix)
    except isolation.ScopeUnreachable as exc:
        raise StorageError('User systemd manager unreachable (%s): scopes %s* not attested; reservation kept'
                           % (exc, prefix)) from None
    for name in names:
        stop_child({'scope': name})


def stop_child(child):
    """Stop what a dead attempt left running, from its `runs.child` record (K7, PB8): the systemd
    scope of a Marker child, the container of a tool (the pid is handled by the caller). An
    unreachable user manager or Docker is no proof that the child is gone: the reconciliation is
    not attested, the reservation is kept and the error is visible."""
    scope = child.get('scope')
    if scope:
        if not isinstance(scope, str) or not isolation.UNIT.fullmatch(scope.removesuffix('.scope')):
            raise StorageError('Malformed child record')
        try:
            state = isolation.stop_scope(scope)
        except isolation.ScopeUnreachable as exc:
            raise StorageError('User systemd manager unreachable (%s): stop of %s not attested; reservation kept'
                               % (exc, scope)) from None
        if state != 'absent':
            raise StorageError('Previous conversion scope still active: %s; reservation kept' % scope)
    container = child.get('container')
    if container:
        if not isinstance(container, str) or not isolation.UNIT.fullmatch(container):
            raise StorageError('Malformed child record')
        from .engine.toolchain import remove_container
        if not remove_container(container):
            raise StorageError('Previous conversion container not removed (Docker unreachable?): %s; '
                               'reservation kept' % container)


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
        if kind == "child":
            child = {"pid": event["pid"]} if event.get("pid") else {}     # no pid: the scope before the start
            if event.get("pid") and (ident := process_identity(event["pid"])):
                child["start"] = ident["start"]
            if event.get("scope"):
                child["scope"] = event["scope"]
            self.child, changed = child, True
        elif kind == "tool" and event.get("container"):
            self.child, changed = {"container": event["container"]}, True
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


def take_datalab_key(environ=None):
    """The Datalab key of the worker: `DATALAB_API_KEY` taken out of the environment of the
    process (so no child can inherit it), or None."""
    from .engine import datalab
    value = (os.environ if environ is None else environ).pop(datalab.KEY_NAME, None)
    return datalab.Key(value) if value else None


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


def engine_request(options, attempt, fmt=None, cap_usd="0"):
    """The request of the engine for the stored options of a run (`services.stored_options`: a
    run of schema 2 maps `force_ocr` to `ocr: force`; `strip_existing_ocr` has no equivalent and
    is dropped, D3 of MFL032: an existing OCR layer is set aside by the engine itself). With `fmt`
    the options are validated again by the routes of the engine (EngineError when refused); the
    Datalab route gets the confirmed transit of the job and the cap of the settings."""
    opts = stored_options(options)
    if fmt is not None:
        opts = routes.job_options(fmt, opts)
    request = {"route": opts.get("route"), "ocr": opts.get("ocr", "auto"), "lang": opts.get("lang"), "attempt": attempt}
    if fmt is not None and routes.resolve(fmt, opts["route"])[0] == routes.DATALAB:
        request["datalab"] = {"mode": opts["datalab_mode"], "transit_confirmed": opts["transit_confirmed"],
                              "cap_usd": cap_usd}
    return request


def summarize(result, inventory):
    """The bounded summary of an engine result that the manifest carries (K9, PB9): every count,
    at most MAX_ISSUES problems (the incomplete ones first, as the engine lists them), the page
    classes without the per-page records, the corrections counted only."""
    summary = {k: result[k] for k in ("schema", "status", "attempt", "started_at", "source", "options", "limits",
                                      "problem_codes", "problem_counts", "components", "timings", "transit", "cost")
               if k in result}
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
    def __init__(self, settings):
        self.settings = settings
        self.service = Service(settings)
        self.storage = self.service.storage
        self.token = uuid.uuid4().hex
        self.stop = False
        self.lease = None
        self._worker_locked = False
        self._ready_at = None
        self.converters = None           # availability per converter, from `ready`
        self.readiness = {"ready": False, "message": "Worker starting", "converters": None}
        self.datalab_key = take_datalab_key()

    def datalab_state(self):
        """Datalab can run here with the key of the worker and a cap above 0."""
        if self.datalab_key is None:
            return {"available": False, "code": Code.DATALAB_KEY_MISSING.value}
        if Decimal(self.settings.datalab_cap_usd) <= 0:
            return {"available": False, "code": Code.DATALAB_CAP.value}
        return {"available": True, "code": None}

    def ready(self):
        """The availability of every converter on this machine (`engine.availability`: the
        toolchain image, Bubblewrap, the systemd user scope and the verified weights of Marker;
        Datalab from the key of the worker and the cap), checked again after READY_SECONDS; the
        engine verifies the weights again before every Marker child. Returns it."""
        now = time.monotonic()
        if self._ready_at is not None and now - self._ready_at < READY_SECONDS:
            return self.converters
        state = availability(model_root=self.settings.model_root)
        state[routes.DATALAB] = self.datalab_state()
        missing = sorted('%s (%s)' % (name, s['code']) for name, s in state.items() if not s['available'])
        self.converters, self._ready_at = state, now
        self.readiness = {"ready": True, "converters": state,
                          "message": "Local worker ready" + ("; routes unavailable: " + ", ".join(missing) if missing else "")}
        return state

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

    def reconcile(self, run_id=None):
        with self.exclusive():
            self._reconcile(run_id)

    def _reconcile(self, run_id):
        from .capacity import recover, live_workers
        rows = [self.storage.get(r['id']) for r in self.storage.list()
                if run_id is None or r['id'] == run_id]
        with self.storage.lock('admission.lock'):
            live = live_workers(self.storage)
        selected = []
        # Reap identified orphan children before releasing their dead commitments.
        # Live commitments from independent operations remain locked and reserved.
        for row in rows:
            if row['attempt_id'] in live and row['token'] != self.token:
                continue
            selected.append(row)
            if row['state'] not in {'running', 'publishing'}:
                continue
            if row.get('scope_recorded') != 0:
                # 1: the attempt recorded a scope before starting it; NULL: an attempt of a previous
                # version, without the indicator. 0: no scope was ever started, nothing to ask
                stop_attempt_scopes(row['attempt_id'])
            if row['child'] and json.loads(row['child']):
                old = json.loads(row['child'])
                stop_child(old)
                now = process_identity(old['pid']) if old.get('pid') else None
                if now and now['start'] == old.get('start') and now['state'] != 'Z':
                    os.kill(old['pid'], signal.SIGKILL)
                    deadline = time.monotonic() + self.settings.grace_seconds
                    while time.monotonic() < deadline:
                        now = process_identity(old['pid'])
                        if not now or now['start'] != old['start'] or now['state'] == 'Z':
                            break
                        time.sleep(0.1)
                    else:
                        raise StorageError('Previous conversion process still alive')
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
        with self.storage.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT id FROM runs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
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
            # a scope is named here before it starts: the indicator of the attempt becomes 1 in the
            # same update, and never goes back (a trigger of 003.sql)
            conn.execute("UPDATE runs SET lease=?,child=?,updated=?,progress=COALESCE(?,progress),"
                         "scope_recorded=CASE WHEN ? THEN 1 ELSE scope_recorded END WHERE id=? AND token=?",
                         (time.time() + self.settings.lease_seconds, json.dumps(child, allow_nan=False) if child else None, time.time(),
                          json.dumps(progress, allow_nan=False) if progress else None,
                          bool(child and child.get('scope')), row["id"], self.token))
        if self.stop:
            return 'interrupted'
        if self.lease:
            self.lease.check(local_only=True)
        return bool(current["cancel"])

    def commitment(self, row):
        return self.storage.reservation('worker', (2 * self.settings.output_mib +
                    self.settings.log_mib + 1) * 1048576, row['attempt_id'],
                    ('attempts/' + row['attempt_id'], 'results/' + row['attempt_id']))

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
                    state = 'cancelled' if current['cancel'] else 'interrupted' if self.stop else 'failed'
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
            request = engine_request(options, row['attempt_id'], fmt, self.settings.datalab_cap_usd)
        except EngineError as exc:          # validated by the API too: a job of another version
            raise StorageError('Conversion refused: %s; source kept' % exc.code.value) from None
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
        result = convert(self.storage.root / row['source_path'], scratch / 'engine-out', request, limits,
                         work=scratch / 'engine-work', observer=observer, name=source_name(row.get('name'), row['source_path']),
                         model_root=self.settings.model_root, datalab_key=self.datalab_key)
        if observer.error is not None:
            raise observer.error
        if result.get('status') != 'complete':
            raise StorageError(engine_message((result.get('error') or {}).get('code', 'engine_failed'), self.settings))
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
                             tick=lambda child, progress: self.tick(row, child, progress), deadline=60, label='raster')
            if raster['exit'] == 0 and not raster['reason']:
                results = validate_raster(read_json(raster.get('dir', scratch), 'raster-result.json'), candidates, self.settings)
                for item in artifacts:
                    item.update(results.get(item['path'], {}))
            else:
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
            self.reconcile()
            if self.stop:
                return
            probe = self.storage.root / 'staging' / ('worker-' + self.token)
            make_directory(self.storage.root, 'staging/' + probe.name)
            with self.storage.reservation('worker', (2 * self.settings.output_mib + 1) * 1048576,
                                          self.token, ('staging/' + probe.name,)):
                receipt = execute(self.settings, probe, 'markdown_for_llms.engines.probe', deadline=20,
                                  memory_mib=256, label='probe')
            if receipt['exit'] or receipt['reason']:
                self.heartbeat({"ready": False, "message": "Isolation unavailable; no fallback", "converters": None})
                raise StorageError('Isolation readiness failed; diagnostic retained')
            while not self.stop:
                try:
                    self.ready()
                    self.heartbeat()
                except (ValueError, OSError, ImportError) as exc:
                    self.heartbeat({"ready": False, "message": "Conversion engine unavailable: " + str(exc)[:120],
                                    "converters": None})
                    if once:
                        raise StorageError("Conversion engine unavailable: " + str(exc)[:120]) from None
                    time.sleep(self.settings.poll_seconds)
                    continue
                row = self.claim()
                if row:
                    self.process(row)
                if once:
                    break
                time.sleep(self.settings.poll_seconds)


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
