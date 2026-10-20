"""Durable single-host worker with explicit retries and publication fencing."""
from pathlib import Path
from contextlib import contextmanager
from dataclasses import replace
import json
import os
import signal
import sqlite3
import logging
import time
import uuid

from .files import identify, make_directory, open_file, read_json, rename, sync_directory, write_json, sync_tree, corruption, relative, IntegrityError, MetadataError, decode_json, publication_barrier, directory, FileSafetyError, json_bytes
from .progress import finishing
from .references import inspect
from .runtime import check_runtime
from .sandbox import execute, process_identity
from .services import Service, ReadUnavailable, stored_manifest, publication_budget
from .storage import StorageError, CapacityError


class Worker:
    def __init__(self, settings):
        self.settings = settings
        self.service = Service(settings)
        self.storage = self.service.storage
        self.token = uuid.uuid4().hex
        self.stop = False
        self.lease = None
        self._worker_locked = False

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

    def heartbeat(self, readiness="ready"):
        with self.storage.connect() as conn:
            conn.execute("INSERT INTO worker VALUES (1,?,?,?) ON CONFLICT(id) DO UPDATE SET token=excluded.token,heartbeat=excluded.heartbeat,readiness=excluded.readiness", (self.token, time.time(), readiness))

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
            if row['child'] and json.loads(row['child']):
                old = json.loads(row['child'])
                now = process_identity(old['pid'])
                if now and now['start'] == old['start'] and now['state'] != 'Z':
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
            conn.execute("UPDATE runs SET state='running',token=?,lease=?,updated=? WHERE id=? AND state='queued'", (self.token, time.time() + self.settings.lease_seconds, time.time(), row["id"]))
        return self.storage.get(row["id"])

    def tick(self, row, child, progress=None):
        """Renew the lease in the same update that records the child and, when given,
        the progress snapshot; without one the last snapshot is kept."""
        self.heartbeat()
        with self.storage.connect() as conn:
            current = conn.execute("SELECT token,cancel FROM runs WHERE id=?", (row["id"],)).fetchone()
            if current["token"] != self.token:
                return True
            conn.execute("UPDATE runs SET lease=?,child=?,updated=?,progress=COALESCE(?,progress) WHERE id=? AND token=?",
                         (time.time() + self.settings.lease_seconds, json.dumps(child, allow_nan=False) if child else None, time.time(),
                          json.dumps(progress, allow_nan=False) if progress else None, row["id"], self.token))
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
        runtime = check_runtime(self.settings)
        write_json(scratch, 'request.json', {'options': json.loads(row['options']),
                   'max_pages': self.settings.max_pages, 'threads': self.settings.threads})
        receipt = execute(self.settings, scratch, self.storage.root / row['source_path'],
                          tick=lambda child, progress: self.tick(row, child, progress),
                          deadline=self.settings.timeout_seconds or None, stall=self.settings.stall_seconds)
        if receipt['reason'] == 'stalled':
            raise StorageError('No conversion progress for %d s (MDF_STALL_SECONDS); attempt stopped, '
                               'source kept, retry available' % self.settings.stall_seconds)
        if receipt['reason'] == 'timeout':
            raise StorageError('Conversion exceeded %d s (MDF_TIMEOUT_SECONDS); attempt stopped, '
                               'source kept, retry available' % self.settings.timeout_seconds)
        if receipt['reason'] or receipt['exit'] != 0:
            raise StorageError(receipt['reason'] or 'Engine process failed; bounded diagnostic log retained')
        self.tick(row, None, finishing(receipt.get('progress')))
        self.publish(row, scratch, runtime, receipt)

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
        if engine.get('issues'):
            checks['issues'].append({'code': 'incomplete', 'reason': 'Engine reported conversion errors'})
            if checks['bundle'] != 'invalid':
                checks['bundle'] = 'incomplete'
        artifacts = [dict(item, id=uuid.uuid4().hex, gallery=False) for item in inventory]
        candidates = [item for item in inventory if item['path'] != 'document.md']
        if candidates:
            write_json(scratch, 'raster-request.json', dict(artifacts=candidates,
                       max_pixels=self.settings.max_pixels, max_dimension=self.settings.max_dimension))
            raster = execute(replace(self.settings, memory_mib=512), scratch,
                             module='markdown_for_llms.engines.raster', label='raster',
                             tick=lambda child, progress: self.tick(row, child, progress), deadline=60)
            if raster['exit'] == 0 and not raster['reason']:
                results = validate_raster(read_json(scratch, 'raster-result.json'), candidates, self.settings)
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
        manifest = dict(schema=1, run_id=row["id"], attempt_id=row["attempt_id"], document_id=row["document_id"],
                        source={"sha256": row["source_sha256"], "bytes": row["source_bytes"]},
                        state="completed", bundle=checks["bundle"], quality="unverified", artifacts=artifacts,
                        checks=checks, engine=engine, runtime=runtime, config=self.settings.public(),
                        elapsed_seconds=receipt["elapsed_seconds"],
                        limits=["Experimental conversion; formulas, OCR, reading order and semantic asset completeness require manual verification"])
        body = json_bytes(manifest)
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
                receipt = execute(replace(self.settings, memory_mib=256), probe,
                                  module='markdown_for_llms.engines.probe', deadline=10)
            if receipt['exit'] or receipt['reason']:
                self.heartbeat('Isolamento non disponibile; nessun fallback')
                raise StorageError('Isolation readiness failed; diagnostic retained')
            while not self.stop:
                try:
                    check_runtime(self.settings, verify_models=False)
                    self.heartbeat()
                except (ValueError, OSError, ImportError) as exc:
                    self.heartbeat("runtime unavailable: " + type(exc).__name__)
                    if once:
                        raise StorageError("ML runtime unavailable") from None
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
