"""Shared application operations for CLI and HTTP."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import fcntl
import json
import math
import os
import time
import uuid
import zipfile

from . import legacy_identity
from .files import digest, identify, open_file, read_json, relative, rename, sync_directory, corruption, make_directory, write_json, IntegrityError, MetadataError, decode_json, FileSafetyError, json_bytes
from .engine import routes
from .engine.codes import EngineError
from .engine.formats import EXTENSIONS, detect, extension_of
from .engine.progress import STAGES
from .storage import OWNER, Storage, StorageError, CapacityError

TERMINAL = {"completed", "failed", "interrupted", "cancelled"}
# The texts of the refusals that carry a closed code of the engine or of its routes (MFL032 C2):
# the API returns the code next to the text.
REFUSALS = {
    "format_unsupported": "Format not supported: the extension of the file name is not in the conversion matrix",
    "format_mismatch": "The content of the file does not match the extension of its name",
    "input_unreadable": "The file cannot be read (damaged or truncated?)",
    "route_invalid": "This format has no such route",
    "route_option_unsupported": "The route does not accept this option",
    "transit_not_confirmed": "The Datalab route sends the document out of this computer: confirm the transit first",
    "invalid_setting": "Invalid job option",
    "route_unavailable": "The route is not available on this machine",
    "route_not_in_product": "The marker-server route is available in mdf convert and mdf bench only, not in the archive",
}


class Refused(StorageError):
    """A request refused with a closed code of the engine or of its routes."""

    def __init__(self, code, message=None):
        super().__init__(message or REFUSALS.get(code, "Request refused"))
        self.code = code


def refused(exc):
    return Refused(exc.code.value)


def source_format(source_path):
    """The format of an archived source, from its name `sources/<token>.<ext>` (MFL032 C2;
    `sources/<token>.pdf` in the archives of previous versions)."""
    fmt = EXTENSIONS.get(extension_of(source_path))
    if fmt is None:
        raise IntegrityError("Archived source of unknown format")
    return fmt


def stored_options(raw):
    """The options of a stored run in the shape of schema 3. A run of schema 2 has the options of
    Marker 1: `force_ocr` becomes `ocr: force`, `strip_existing_ocr` has no equivalent (D3 of
    MFL032: the engine sets an existing OCR layer aside by itself)."""
    value = json.loads(raw) if isinstance(raw, str) else raw
    if legacy_identity.v2_options(value):
        return {"route": None, "ocr": "force" if value["force_ocr"] else "auto", "lang": None,
                "datalab_mode": None, "transit_confirmed": False}
    if not isinstance(value, dict):
        raise IntegrityError("Malformed stored options")
    return value


class Service:
    def __init__(self, settings):
        self.settings = settings
        self.storage = Storage(settings)

    def previous(self, key):
        if key is None:
            return None
        if not isinstance(key, str) or len(key) > 128 or not key:
            raise StorageError("Invalid idempotency key")
        with self.storage.connect() as conn:
            row = conn.execute("SELECT r.id,r.options,d.source_path,d.source_sha256,d.source_bytes FROM runs r JOIN documents d ON d.id=r.document_id WHERE idempotency_key=?", (key,)).fetchone()
        return dict(row) if row else None

    @contextmanager
    def upload(self, name, overrides=None, key=None):
        """Receive one file of the conversion matrix: its format comes from the extension of
        `name` and is confirmed by its signature in `finish` (K3 of MFL032); the options are
        validated by the routes of the engine before any byte is stored."""
        if not isinstance(name, str) or not name or '\0' in name:
            raise Refused('format_unsupported')
        ext = extension_of(name)
        if ext not in EXTENSIONS:
            raise Refused('format_unsupported')
        try:
            opts = routes.job_options(EXTENSIONS[ext], overrides)
        except EngineError as exc:
            raise refused(exc) from None
        previous = self.previous(key)
        receiver = Upload(self, name, opts, key, ext)
        if previous:
            # Validate the complete replay without making another disk copy.
            receiver.previous = previous
            yield receiver
            return
        token = receiver.token
        stage = 'staging/upload-' + token
        dest = 'sources/' + token + '.' + ext
        with self.storage.reservation('upload', self.settings.upload_mib * 1048576 + 65536,
                                      token, (stage, dest)):
            make_directory(self.storage.root, stage)
            with open_file(self.storage.root, stage + '/body', 'xb') as stream:
                receiver.stream = stream
                yield receiver

    def submit(self, stream, name, overrides=None, key=None):
        with self.upload(name, overrides, key) as receiver:
            while chunk := stream.read(65536):
                receiver.write(chunk)
            return receiver.finish()

    def read_failure(self, run_id, exc):
        if corruption(exc):
            self.invalidate(run_id, 'Source, manifest or artifact missing/altered/unsafe')
            raise StorageError('Bundle invalid: source, manifest or artifact missing/altered/unsafe') from exc
        raise ReadUnavailable('Read temporarily unavailable; retry without converting again') from exc

    def get(self, run_id):
        row = self.storage.get(run_id)
        result = {k: row[k] for k in ("id", "document_id", "attempt_id", "retry_of", "name", "state", "bundle", "quality", "created", "updated", "error")}
        result["options"] = stored_options(row["options"])
        result["format"] = source_format(row["source_path"])
        try:
            result["manifest"] = stored_manifest(row['manifest'], row) if row['manifest'] else None
        except IntegrityError:
            result['manifest'] = None
            result['bundle'] = 'invalid'
            result['error'] = ((row['error'] + '; ') if row['error'] else '') + 'Malformed stored manifest; source and result preserved'
        result["download_available"] = row["state"] == "completed" and row["bundle"] in {"ready", "unchecked"}
        result['download_available'] &= result['bundle'] in {'ready', 'unchecked'}
        if row['state'] == 'publishing':
            # A status read is not an admission. Inspect only the worker lock;
            # never scan the archive or wait on admission.lock during polling.
            deferred = bool(row['error'])
            if not deferred:
                try:
                    with open_file(self.storage.root, 'worker.lock') as stream:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        deferred = True
                except FileNotFoundError:
                    deferred = True
                except BlockingIOError:
                    pass
                except (OSError, StorageError, FileSafetyError):
                    result['diagnosis'] = 'Worker availability temporarily unknown'
            if deferred:
                result['error'] = result['error'] or 'Publication pending; worker recovery deferred'
                result['recovery'] = 'Restore capacity/I/O if unavailable, then restart the worker to reconcile this same run; no new conversion'
        states = {'queued': 'Queued', 'running': 'Converting', 'publishing': 'Publishing',
                  'completed': 'Completed', 'failed': 'Failed', 'interrupted': 'Interrupted', 'cancelled': 'Cancelled'}
        bundles = {'pending': 'bundle pending', 'ready': 'bundle available; references checked in the profile',
                   'unchecked': 'bundle available; references not completely checked',
                   'incomplete': 'missing asset or known error; bundle incomplete', 'invalid': 'bundle invalid'}
        result['message'] = states[row['state']] + ' — ' + bundles[result['bundle']] + ' — quality not verified'
        if row['state'] == 'interrupted':
            result['message'] += ' — source kept; retry available'
        result['progress'] = progress_view(row['state'], row.get('progress'), self.settings.stall_seconds)
        return result

    def list(self):
        return [self.get(row["id"]) for row in self.storage.list()]

    def retry(self, run_id, overrides=None):
        row = self.storage.get(run_id)
        if row["state"] not in TERMINAL:
            raise StorageError("Only terminal runs can be retried")
        opts = stored_options(row["options"])
        # the confirmation of a transit to Datalab is for one sending: a retry confirms it again
        opts["transit_confirmed"] = False
        opts.update(overrides or {})
        try:
            opts = routes.job_options(source_format(row["source_path"]), opts)
        except EngineError as exc:
            raise refused(exc) from None
        new_id, attempt = uuid.uuid4().hex, uuid.uuid4().hex
        with self.storage.reservation('control', 65536), self.storage.connect() as conn:
            now = time.time()
            conn.execute("INSERT INTO runs (id,document_id,retry_of,attempt_id,options,state,created,updated,owner,scope_recorded) VALUES (?,?,?,?,?,'queued',?,?,?,0)", (new_id, row["document_id"], run_id, attempt, json.dumps(opts, allow_nan=False), now, now, OWNER))
        return self.get(new_id)

    def cancel(self, run_id):
        with self.storage.connect() as conn:
            conn.execute("UPDATE runs SET cancel=1,state=CASE WHEN state='queued' THEN 'cancelled' ELSE state END,updated=? WHERE id=? AND state IN ('queued','running')", (time.time(), run_id))
        return self.get(run_id)

    def invalidate(self, run_id, reason):
        with self.storage.connect() as conn:
            conn.execute("UPDATE runs SET bundle='invalid',error=?,updated=? WHERE id=?", (reason, time.time(), run_id))

    def verified_files(self, run_id):
        row = self.storage.get(run_id)
        if row["state"] != "completed" or row["bundle"] not in {"ready", "unchecked"}:
            raise StorageError("Ordinary bundle unavailable for this state")
        try:
            manifest = stored_manifest(row['manifest'], row)
        except IntegrityError as exc:
            self.read_failure(run_id, exc)
        try:
            paths = [("source/document." + extension_of(row["source_path"]), row["source_path"],
                      {"sha256": row["source_sha256"], "bytes": row["source_bytes"]})]
            for artifact in manifest["artifacts"]:
                paths.append(("result/" + relative(artifact["path"]), row["result_path"] + "/original/" + artifact["path"], artifact))
            if read_json(self.storage.root, row['result_path'] + '/manifest.json') != manifest:
                raise IntegrityError('Publication manifest changed')
            for _, path, expected in paths:
                actual = identify(self.storage.root, path)
                if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
                    raise IntegrityError("Artifact identity mismatch")
        except MetadataError as exc:
            self.read_failure(run_id, IntegrityError('Malformed stored manifest'))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.read_failure(run_id, exc)
        return row, manifest, paths

    def bundle(self, run_id):
        row, manifest, paths = self.verified_files(run_id)
        token = row['attempt_id']
        final_name = 'downloads/' + token + '.zip'
        receipt_name = 'downloads/' + token + '.json'
        name = 'downloads/' + token + '.part'
        with self.storage.lock('zip-' + token + '.lock'):
            with self.storage.lock('admission.lock'):
                from .capacity import recover
                recover(self.storage)
            # Original identities are checked on every delivery, including cache hits.
            row, manifest, paths = self.verified_files(run_id)
            if (self.storage.root / final_name).exists():
                try:
                    expected = read_json(self.storage.root, receipt_name)
                    actual = identify(self.storage.root, final_name)
                    if actual != expected:
                        raise IntegrityError('Cached ZIP identity mismatch')
                    with open_file(self.storage.root, final_name) as stream:
                        with zipfile.ZipFile(stream) as archive:
                            if archive.testzip() is not None:
                                raise IntegrityError('Cached ZIP CRC mismatch')
                except (OSError, ValueError, zipfile.BadZipFile) as exc:
                    if isinstance(exc, OSError) and not corruption(exc):
                        raise ReadUnavailable('ZIP read temporarily unavailable; retry') from exc
                    raise StorageError('Cached ZIP unavailable; archive preserved') from exc
                return self.storage.root / final_name
            # ZIP_STORED has a calculable upper bound, including headers and manifest.
            manifest_bytes = json_bytes(manifest)
            required = sum(p[2]['bytes'] for p in paths) + len(manifest_bytes) + sum(
                512 + 4 * len(p[0].encode()) for p in paths) + 65536
            with self.storage.reservation('zip', required, token, (name, final_name, receipt_name)) as lease:
                with open_file(self.storage.root, name, 'xb') as output:
                    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
                        for member, path, expected in paths:
                            try:
                                with open_file(self.storage.root, path) as stream:
                                    h, size = hashlib.sha256(), 0
                                    with archive.open(member, 'w', force_zip64=True) as target:
                                        while chunk := stream.read(65536):
                                            size += len(chunk)
                                            if size > expected['bytes']:
                                                raise IntegrityError('Artifact grew during download')
                                            h.update(chunk)
                                            target.write(chunk)
                                    if h.hexdigest() != expected['sha256'] or size != expected['bytes']:
                                        raise IntegrityError('Artifact changed during download')
                            except (OSError, ValueError) as exc:
                                self.read_failure(run_id, exc)
                        archive.writestr('manifest.json', manifest_bytes)
                    output.flush()
                    os.fsync(output.fileno())
                with open_file(self.storage.root, name) as check:
                    with zipfile.ZipFile(check) as archive:
                        if archive.testzip() is not None:
                            raise StorageError('ZIP integrity check failed')
                lease.check()
                # Publish receipt first: crash before rename leaves a recoverable part.
                value = dict(identify(self.storage.root, name), path=final_name)
                if (self.storage.root / receipt_name).exists():
                    (self.storage.root / receipt_name).unlink()
                write_json(self.storage.root, receipt_name, value)
                rename(self.storage.root, name, final_name)
            return self.storage.root / final_name


SNAPSHOT_KEYS = {'stage', 'estimate_percent', 'phase', 'current', 'total', 'updated', 'last_advance'}


def progress_view(state, stored, stall_seconds, now=None):
    """Estimated engine progress of a run: separate from its state, never an error.
    None while queued, once completed, or when no valid snapshot was recorded."""
    if state in ('queued', 'completed'):
        return None
    live = state in ('running', 'publishing')
    if stored is None:
        if not live:
            return None
        value = dict(stage='starting', estimate_percent=0.0, phase=None, current=None, total=None,
                     updated=None, last_advance=None)
    else:
        try:
            value = json.loads(stored)
        except (TypeError, ValueError):
            return None
    if (not isinstance(value, dict) or not SNAPSHOT_KEYS <= value.keys()
            or not isinstance(value['stage'], str) or value['stage'] not in STAGES
            or type(value['estimate_percent']) not in (int, float)
            or not 0 <= value['estimate_percent'] <= 100
            or not (value['phase'] is None or isinstance(value['phase'], str) and len(value['phase']) <= 64)
            or any(not (value[k] is None or type(value[k]) is int and value[k] >= 0) for k in ('current', 'total'))
            or any(not (value[k] is None or type(value[k]) in (int, float) and math.isfinite(value[k]))
                   for k in ('updated', 'last_advance'))):
        return None
    if value['stage'] == 'finishing':
        activity = ('Finishing: image checks and publication', 'Stopped while finishing')
    elif value['phase'] is None:
        activity = ('Starting engine', 'Stopped before the first engine progress')
    else:
        counted = value['phase'] if value['current'] is None or value['total'] is None else (
            '%s %d/%d' % (value['phase'], value['current'], value['total']))
        activity = (counted, 'Stopped during ' + counted)
    idle = None
    if live and value['last_advance'] is not None:
        idle = max(0.0, round((time.time() if now is None else now) - value['last_advance'], 1))
    return dict(estimate_percent=value['estimate_percent'], estimated=True, stage=value['stage'],
                phase=value['phase'], current=value['current'], total=value['total'],
                last_advance=value['last_advance'], idle_seconds=idle, stall_seconds=stall_seconds,
                live=live, label='%s (≈%d%%, estimate)' % (activity[0 if live else 1], value['estimate_percent']))


class ReadUnavailable(StorageError):
    """An intact result may be retried without another conversion."""


def publication_budget(manifest_body):
    return 4 * len(manifest_body.encode()) + 131072


def stored_manifest(body, row=None):
    """Validate the stored shape before consuming manifest fields at read boundaries."""
    try:
        value = decode_json(body)
    except (MetadataError, TypeError, AttributeError) as exc:
        raise IntegrityError('Malformed stored manifest') from exc
    if (not isinstance(value, dict) or not isinstance(value.get('artifacts'), list)
            or not 1 <= len(value['artifacts']) <= 4096
            or not isinstance(value.get('bundle'), str)
            or value['bundle'] not in {'ready', 'unchecked', 'incomplete', 'invalid'}
            or type(value.get('schema')) is not int or value['schema'] != 1
            or value.get('state') != 'completed' or value.get('quality') != 'unverified'
            or any(not isinstance(value.get(k), str) for k in ('run_id', 'attempt_id', 'document_id'))
            or not isinstance(value.get('source'), dict)
            or type(value['source'].get('bytes')) is not int or value['source']['bytes'] < 0
            or not isinstance(value['source'].get('sha256'), str)
            or len(value['source']['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in value['source']['sha256'])
            or not isinstance(value.get('checks'), dict)
            or not isinstance(value['checks'].get('issues'), list)
            or any(not isinstance(issue, dict) for issue in value['checks']['issues'])
            or value['checks'].get('bundle') != value['bundle']
            or not isinstance(value.get('engine'), dict)
            or value['engine'].get('inventory_complete') is not True
            or not isinstance(value['engine'].get('inventory'), list)
            or not isinstance(value.get('runtime'), dict)
            or not isinstance(value.get('config'), dict)
            or type(value.get('elapsed_seconds')) not in {int, float}
            or value['elapsed_seconds'] < 0
            or not isinstance(value.get('limits'), list)):
        raise IntegrityError('Malformed stored manifest')
    if row is not None and (any(value[k] != row[r] for k, r in
            [('run_id', 'id'), ('attempt_id', 'attempt_id'), ('document_id', 'document_id')])
            or value['source'] != {'bytes': row['source_bytes'], 'sha256': row['source_sha256']}):
        raise IntegrityError('Stored manifest identity mismatch')
    paths, ids = set(), set()
    for item in value['artifacts']:
        if (not isinstance(item, dict) or not isinstance(item.get('path'), str)
                or not isinstance(item.get('id'), str) or item['id'] in ids
                or item['path'] in paths or type(item.get('bytes')) is not int
                or item['bytes'] < 0 or type(item.get('gallery')) is not bool
                or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise IntegrityError('Malformed stored artifact identity')
        try:
            relative(item['path'])
        except FileSafetyError as exc:
            raise IntegrityError('Unsafe stored artifact path') from exc
        ids.add(item['id'])
        paths.add(item['path'])
    if 'document.md' not in paths:
        raise IntegrityError('Missing Markdown inventory')
    if value['engine']['inventory'] != [{k: item[k] for k in ('path', 'bytes', 'sha256')}
                                       for item in value['artifacts']]:
        raise IntegrityError('Stored engine inventory mismatch')
    return value


class Upload:
    """One bounded ingestion path for CLI and the HTTP request body."""
    def __init__(self, service, name, opts, key, ext):
        self.service, self.name, self.opts, self.key, self.ext = service, name, opts, key, ext
        self.token = uuid.uuid4().hex
        self.stream, self.previous = None, None
        self.hash, self.size = hashlib.sha256(), 0

    def write(self, chunk):
        self.size += len(chunk)
        if self.size > self.service.settings.upload_mib * 1048576:
            raise StorageError('Upload limit exceeded (MDF_UPLOAD_MIB)')
        self.hash.update(chunk)
        if self.stream:
            self.stream.write(chunk)

    def finish(self):
        sha = self.hash.hexdigest()
        service = self.service
        stage = 'staging/upload-' + self.token
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            # K3 of MFL032: extension and signature on the whole staged file (the central
            # directory of a ZIP package is at its end), before the admission
            try:
                detect(service.storage.root / stage / 'body', self.name)
            except EngineError as exc:
                raise refused(exc) from None
        with service.storage.lock('admission.lock'):
            previous = service.previous(self.key)
            if previous:
                # a replay: the same bytes are the same format; the extension of the new name must
                # be the one of the registered source (PC1 of MFL032)
                if (previous['source_sha256'] != sha or previous['source_bytes'] != self.size
                        or stored_options(previous['options']) != self.opts
                        or extension_of(previous['source_path']) != self.ext):
                    raise StorageError('Idempotency key conflicts with different input')
                return service.get(previous['id'])
            if self.stream is None:
                raise StorageError('Idempotent registration disappeared; retry request')
            document_id, run_id, attempt_id = [uuid.uuid4().hex for _ in range(3)]
            dest = 'sources/' + self.token + '.' + self.ext
            with service.storage.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                rename(service.storage.root, stage + '/body', dest)
                now = time.time()
                conn.execute('INSERT INTO documents (id,name,source_path,source_sha256,source_bytes,created,owner) VALUES (?,?,?,?,?,?,?)',
                             (document_id, self.name[:255], dest, sha, self.size, now, OWNER))
                conn.execute("INSERT INTO runs (id,document_id,attempt_id,idempotency_key,options,state,created,updated,owner,scope_recorded) VALUES (?,?,?,?,?,'queued',?,?,?,0)",
                             (run_id, document_id, attempt_id, self.key, json.dumps(self.opts, allow_nan=False), now, now, OWNER))
        return service.get(run_id)
