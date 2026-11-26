"""Shared application operations for CLI and HTTP."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import fcntl
import json
import math
import os
import re
import sqlite3
import time
import uuid
import zipfile

from . import legacy_identity
from .files import digest, identify, open_file, read_json, relative, rename, sync_directory, corruption, make_directory, write_json, IntegrityError, MetadataError, decode_json, FileSafetyError, json_bytes
from .engine import routes
from .engine.codes import Code, EngineError
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
    "transit_not_confirmed": ("This route sends the document out of this computer (to Datalab, or its page images to the "
                              "rented GPU of a GPU preset): confirm the transit first"),
    "invalid_setting": "Invalid job option",
    "route_unavailable": "The route is not available on this machine",
    "route_not_in_product": ("In the archive the marker-server route runs only on a GPU preset of the instance; a local or "
                             "own inference server is for mdf convert and mdf bench only"),
    "server_cap": ("The GPU spending caps of this instance leave no session for this preset: a cap is 0, or the reserve "
                   "left under the caps is less than the minimum reserve of a session (see the estimate)"),
    # what the worker reads of a GPU preset (MFL023 B10, C-DB-NB2.3: its codes in the presets and the estimate)
    "server_key_missing": "A key of the GPU provider of this preset is missing from the env file of the worker",
    "server_key_unsafe": ("The env file of the worker is not a private regular file (mode 0600, not a link), or a key in "
                          "it is unsafe"),
    "server_not_configured": ("No endpoint of servers.json names this GPU preset, or its endpoint is not valid (fields "
                              "fixed by the preset, key_env, the URL read at the provider)"),
    "server_billed_over_bound": ("A reconciliation recorded a bill of this provider over the bound of its windows: the "
                                 "guarantees of a session did not hold, and the provider stays stopped for good"),
    # the conversion presets (MFL023 A5) and the GPU presets they name
    "preset_unknown": "No conversion preset of that name",
    "preset_reserved": "A conversion preset of the instance cannot be changed or removed, and its name is reserved",
    "preset_exists": "A conversion preset of that name exists already",
    "preset_invalid": "Invalid conversion preset",
    # MFL051, the only new code of the tranche A2: `mdf convert --preset` with a preset on a rented GPU
    "preset_needs_archive": ("A conversion preset with a rented GPU runs through the archive, where its session, its "
                             "ledger and its caps are: mdf submit --preset NAME --wait FILE"),
    "gpu_preset_unknown": "No GPU preset of that name",
    "gpu_preset_superseded": "The GPU preset is superseded: it is listed for its measures and never used",
    "gpu_volume_unsupported": "The GPU preset has a global volume, which arrives with a later version",
    "gpu_preset_not_enabled": "The GPU preset is not enabled on this instance",
    "session_orphaned": ("A GPU window of this provider is orphan or not certified: the ledger counts the provider at its "
                         "cap until the window is reconciled with the invoice"),
}
GPU_FINAL = ("closed", "uncertain", "orphan", "stopped")
# The texts of a session that cannot be certified (MFL023 B5, plan r3 §4.8): the Web UI fills them.
UNCERTAIN_TEXT = {
    "unobserved_interval": ("This computer was suspended, or its clock jumped, while the {provider} session was armed "
                            "(about {minutes} min not observed). The worker disarmed {provider} when it resumed and saw the "
                            "GPU at zero at {zero}, but it cannot prove what was spent while it was not watching."),
    "over_window": ("{provider} reached zero at {zero}, after the limit of this session's window ({limit}): its spend "
                    "cannot be bounded by the reserve of the window."),
    "worker_died": ("The worker stopped while the {provider} session was armed. The next start disarmed {provider} and "
                    "saw the GPU at zero, but the spend of the window is known only from the invoice."),
    "zero_missing": ("The worker could not see {provider} at zero within 600 s: the window stays open in the ledger. "
                     "Disarm the endpoint with the first command, then reconcile."),
}
BLOCK_TEXT = ("Until you reconcile this window with the invoice of {provider}, the spend ledger counts {provider} at its "
              "cap ({cap} USD) and no new {provider} session opens. Read on the billing of {provider} the amount for "
              "{date}, then run:")
# The conversion presets (MFL023 A5, D7): how a document is converted. Those of the instance come with it
# (data/conversion-presets.json, provisional until MFL027); a user adds personal ones, in the database with an
# owner, with the same closed fields; a preset names a GPU preset, never the other way round.
CONVERSION_PRESETS = Path(__file__).resolve().parent / "data" / "conversion-presets.json"
PRESET_OPTIONS = ("route", "server_mode", "gpu_preset", "ocr", "datalab_mode")
PRESET_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


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


def job_options(fmt, options=None, settings=None):
    """The normalized options of a job, one function for the API (`Service.upload`, `Service.retry`) and the
    worker (`worker.engine_request`), so the two processes refuse the same requests (MFL023 A5): those of the
    routes of the engine, plus `preset`, the name of the conversion preset the options came from, kept only when
    given (a job without a preset stores no key more, not even null). On the route marker-server (MFL023 B3) the
    GPU preset must be usable by the instance of `settings` (`gpu_availability`; without settings nothing is
    enabled). EngineError with a closed code of the engine, `Refused` with a code of the GPU presets."""
    opts = dict(options or {})
    preset = opts.pop("preset", None)
    if "preset" in (options or {}) and (not isinstance(preset, str) or not PRESET_NAME.fullmatch(preset)):
        raise EngineError(Code.INVALID_SETTING, "preset: the name of a conversion preset")
    result = routes.job_options(fmt, opts)
    if result["route"] == routes.MARKER_SERVER:
        state = gpu_availability(result["gpu_preset"], settings)
        if not state["available"]:
            if state["code"] == Code.SERVER_CAP.value:
                raise EngineError(Code.SERVER_CAP, "the GPU caps of the instance")
            raise Refused(state["code"])
    if preset is not None:
        result["preset"] = preset
    return result


def gpu_availability(name, settings):
    """Whether the instance of `settings` can use the GPU preset `name` (MFL023 B3), {"available", "code"}, in the
    order of the checks: `gpu_preset_unknown`; the code that keeps the preset from any instance (superseded, a global
    volume: `gpu.presets.refusal`); `gpu_preset_not_enabled` (not in MDF_GPU_PRESETS); `server_cap` (a cap of the
    ledger of its provider, or of a session, of 0 USD). What only the worker knows (the endpoint of servers.json, the
    keys of its env file, an orphan window, the minimum reserve) comes with its readiness, never from here: the API
    process reads neither the ledger nor a provider."""
    from decimal import Decimal
    from .gpu import presets as gpu_presets
    table = gpu_presets.load()
    if name not in table:
        return {"available": False, "code": "gpu_preset_unknown"}
    code = gpu_presets.refusal(table[name])
    if code is not None:
        return {"available": False, "code": code}
    if settings is None or name not in settings.gpu_presets_enabled():
        return {"available": False, "code": "gpu_preset_not_enabled"}
    provider_cap = getattr(settings, "gpu_cap_usd_" + table[name]["provider"])
    if Decimal(provider_cap) <= 0 or Decimal(settings.gpu_session_cap_usd) <= 0:
        return {"available": False, "code": Code.SERVER_CAP.value}
    return {"available": True, "code": None}


def check_preset_options(value):
    """The options of a conversion preset, its closed fields (`route`, `server_mode`, `gpu_preset`, `ocr`,
    `datalab_mode`; a missing one is null, `ocr` auto): `marker-server` with a server mode and a GPU preset that can
    be used, those two only with it, a Datalab mode only with Datalab. Refused (`preset_invalid` or the code of the
    GPU preset); the checks of a format are those of the job."""
    if not isinstance(value, dict) or set(value) - set(PRESET_OPTIONS):
        raise Refused("preset_invalid")
    opts = {k: value.get(k) for k in PRESET_OPTIONS}
    if opts["ocr"] is None:
        opts["ocr"] = "auto"
    route = opts["route"]
    if route is not None and route not in routes.CONVERTERS or opts["ocr"] not in routes.OCR_MODES:
        raise Refused("preset_invalid")
    if route == routes.MARKER_SERVER:
        if opts["server_mode"] not in routes.SERVER_MODES or not isinstance(opts["gpu_preset"], str):
            raise Refused("preset_invalid")
        from .gpu import presets as gpu_presets
        table = gpu_presets.load()
        if opts["gpu_preset"] not in table:
            raise Refused("gpu_preset_unknown")
        code = gpu_presets.refusal(table[opts["gpu_preset"]])
        if code is not None:
            raise Refused(code)
    elif opts["server_mode"] is not None or opts["gpu_preset"] is not None:
        raise Refused("preset_invalid")
    if route == routes.DATALAB:
        if opts["datalab_mode"] is not None and opts["datalab_mode"] not in routes.DATALAB_MODES:
            raise Refused("preset_invalid")
    elif opts["datalab_mode"] is not None:
        raise Refused("preset_invalid")
    return opts


def builtin_presets(path=None):
    """The conversion presets of the instance, {name: {provisional, label, description, options}}, checked."""
    try:
        doc = json.loads(Path(path or CONVERSION_PRESETS).read_bytes())
        if set(doc) != {"schema", "presets"} or doc["schema"] != 1 or not isinstance(doc["presets"], dict):
            raise ValueError("schema")
        out = {}
        for name, entry in doc["presets"].items():
            if (not PRESET_NAME.fullmatch(name) or set(entry) != {"provisional", "label", "description", "options"}
                    or type(entry["provisional"]) is not bool
                    or not all(isinstance(entry[k], str) and entry[k] for k in ("label", "description"))):
                raise ValueError("entry")
            out[name] = dict(entry, options=check_preset_options(entry["options"]))
        return out
    except (OSError, ValueError, TypeError, AttributeError, Refused):
        raise IntegrityError("Malformed conversion presets of the instance") from None


PROVIDER_NAMES = {"runpod": "RunPod", "modal": "Modal"}


def gpu_notice(name, preset):
    """The notice of the transit of a GPU preset, as `api.TRANSIT_NOTICE` for Datalab (plan r2 §4.8)."""
    group = preset["gpu"]["group"]
    where = preset["data_center"] or "a region chosen by Modal"
    return ("With this preset the page images leave this computer: they are sent to a GPU rented by the second on %s "
            "(%s, %s) through the relay of the worker, which keeps the key; nothing is stored there after the session. "
            "Send only documents you may share with %s." % (PROVIDER_NAMES[preset["provider"]], where,
                                                          "/".join(group) if isinstance(group, list) else group,
                                                          PROVIDER_NAMES[preset["provider"]]))


GPU_FACTS = ("preset", "provider", "arm_id", "a_cents", "until", "reserve_cents", "bound_cents", "completed", "error")


def gpu_facts(raw):
    """The facts of a run on its GPU session (`runs.gpu`, closed fields written by the worker), or None."""
    try:
        value = json.loads(raw) if raw else None
    except ValueError:
        return None
    return {k: value.get(k) for k in GPU_FACTS} if isinstance(value, dict) else None


def gpu_commands(provider, resource, ledger_path, arm_id=None, stopped=False):
    """The commands that close a GPU window by hand (plan r2 §4.4, r3 §4.7): the reconciliation with the invoice and,
    for a session that lost its zero, the disarm of its endpoint first. Shared by the worker (its heartbeat) and the
    API."""
    runs = str(Path(ledger_path).parent)
    out = ["mdf bench spend reconcile --runs-dir %s --provider %s --orphan%s --manual-cents N"
           % (runs, provider, " --arm %s" % arm_id if arm_id else "")]
    if stopped:
        out.insert(0, "python -I -B scripts/servers/session.py disarm --provider %s --name %s --env-file .env "
                      "--spend-ledger %s" % (provider, resource, ledger_path))
    return out


SESSION_FIELDS = ("arm_id", "gpu_preset", "provider", "state", "reason", "origin", "a_cents", "w_s", "t_arm", "until",
                  "t_zero", "limit_at", "drift_s", "created", "updated")


def session_view(row, ledgers):
    """What the API and the Web UI show of a row of `gpu_sessions`: its closed fields, the total and the cap of its
    provider in the ledger (from the worker), and for a window that cannot be certified, orphan or stopped the texts of
    the block (plan r3 §4.8) and the commands that close it."""
    from datetime import datetime, timezone
    out = {k: row.get(k) for k in SESSION_FIELDS}
    led = (ledgers or {}).get(row["provider"]) or {}
    out.update(ledger_total_cents=led.get("total_cents"), cap_cents=led.get("cap_cents"))
    if row["state"] not in ("uncertain", "orphan", "stopped"):
        return out
    name = {"runpod": "RunPod", "modal": "Modal"}.get(row["provider"], row["provider"])

    def when(t):
        return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if t else "an unknown time"
    reason = row["reason"] if row["reason"] in UNCERTAIN_TEXT else ("zero_missing" if row["state"] == "stopped"
                                                                    else "worker_died")
    cap = led.get("cap_cents")
    out["commands"] = gpu_commands(row["provider"], row["resource"], row["ledger_path"], row["arm_id"],
                                   stopped=row["state"] == "stopped")
    out["texts"] = {"title": "GPU session not certified" if row["state"] == "uncertain" else "GPU window orphan",
                    "reason": UNCERTAIN_TEXT[reason].format(provider=name, minutes=round(abs(row.get("drift_s") or 0) / 60),
                                                            zero=when(row.get("t_zero")), limit=when(row.get("limit_at"))),
                    "block": BLOCK_TEXT.format(provider=name, cap="%.2f" % (float(cap) / 100) if cap else "?",
                                               date="the hours from %s to %s" % (when(row.get("t_arm")),
                                                                                 when(row.get("t_zero") or row.get("until"))))}
    return out


def with_worker(state, worker):
    """The availability of a GPU preset for the instance (`gpu_availability`), and, when the instance allows it, what the
    worker last read (MFL023 B6): an orphan or uncertain window of the provider, the endpoint or the keys missing, a
    reserve under the minimum. The validation of a job stays that of the instance alone, the same in the API and the
    worker (`job_options`)."""
    if state["available"] and isinstance(worker, dict) and worker.get("available") is False:
        return {"available": False, "code": worker.get("code")}
    return state


def preset_availability(options, settings=None):
    """Whether a conversion preset can be used by the archive now, with the code when it cannot: on the GPU route,
    whether the instance can use its GPU preset (`gpu_availability`, MFL023 B3)."""
    if options["route"] == routes.MARKER_SERVER:
        return gpu_availability(options["gpu_preset"], settings)
    return {"available": True, "code": None}


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
            opts = job_options(EXTENSIONS[ext], overrides, self.settings)
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
        result["gpu"] = gpu_facts(row.get("gpu"))              # MFL023 B6: the closed facts of a run on its GPU session
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
            opts = job_options(source_format(row["source_path"]), opts, self.settings)
        except EngineError as exc:
            raise refused(exc) from None
        new_id, attempt = uuid.uuid4().hex, uuid.uuid4().hex
        with self.storage.reservation('control', 65536), self.storage.connect() as conn:
            now = time.time()
            conn.execute("INSERT INTO runs (id,document_id,retry_of,attempt_id,options,state,created,updated,owner,scope_recorded) VALUES (?,?,?,?,?,'queued',?,?,?,0)", (new_id, row["document_id"], run_id, attempt, json.dumps(opts, allow_nan=False), now, now, OWNER))
        return self.get(new_id)

    # --- conversion presets (MFL023 A5) ---------------------------------------------------------------

    def presets(self, worker=True):
        """The conversion presets of the instance and of the user, the default of the user, and the GPU presets with
        whether this instance enables them; each with its availability (`preset_availability`, `gpu_availability`:
        the checks of the instance; the readiness of the worker says the rest, unless `worker` is False)."""
        from .gpu import presets as gpu_presets
        with self.storage.connect() as conn:
            rows = conn.execute("SELECT name, options, created, updated FROM conversion_presets WHERE owner=? "
                                "ORDER BY name", (OWNER,)).fetchall()
            default = conn.execute("SELECT default_preset FROM user_settings WHERE owner=?", (OWNER,)).fetchone()
        worker = ((self.worker_gpu() or {}).get("presets") or {}) if worker else {}

        def state(options):
            out = preset_availability(options, self.settings)
            if options["route"] == routes.MARKER_SERVER:
                out = with_worker(out, worker.get(options["gpu_preset"]))
            return out
        items = [dict(entry, name=name, builtin=True, **state(entry["options"])) for name, entry in builtin_presets().items()]
        for row in rows:
            options = json.loads(row["options"])
            items.append({"name": row["name"], "builtin": False, "options": options, "created": row["created"],
                          "updated": row["updated"], **state(options)})
        enabled = self.settings.gpu_presets_enabled()
        gpus = [dict(gpu_presets.public(name, p), enabled=name in enabled,
                     **with_worker(gpu_availability(name, self.settings), worker.get(name)))
                for name, p in gpu_presets.load().items()]
        return {"presets": items, "default": default["default_preset"] if default else None, "gpu_presets": gpus}

    def preset(self, name, worker=True):
        """One conversion preset, of the instance or of the user; `preset_unknown`. `worker`: as in `presets`."""
        for item in self.presets(worker)["presets"]:
            if item["name"] == name:
                return item
        raise Refused("preset_unknown")

    def add_preset(self, name, options):
        if not isinstance(name, str) or not PRESET_NAME.fullmatch(name):
            raise Refused("preset_invalid")
        if name in builtin_presets():
            raise Refused("preset_reserved")
        opts = json.dumps(check_preset_options(options), sort_keys=True)
        try:
            with self.storage.reservation('control', 65536), self.storage.connect() as conn:
                now = time.time()
                conn.execute("INSERT INTO conversion_presets (id,owner,name,options,created,updated) VALUES (?,?,?,?,?,?)",
                             (uuid.uuid4().hex, OWNER, name, opts, now, now))
        except sqlite3.IntegrityError:
            raise Refused("preset_exists") from None
        return self.preset(name)

    def replace_preset(self, name, options):
        if name in builtin_presets():
            raise Refused("preset_reserved")
        opts = json.dumps(check_preset_options(options), sort_keys=True)
        with self.storage.reservation('control', 65536), self.storage.connect() as conn:
            changed = conn.execute("UPDATE conversion_presets SET options=?, updated=? WHERE owner=? AND name=?",
                                   (opts, time.time(), OWNER, name)).rowcount
        if not changed:
            raise Refused("preset_unknown")
        return self.preset(name)

    def remove_preset(self, name):
        """Removes a personal preset; when it was the default of the user, the user has no default any more."""
        if name in builtin_presets():
            raise Refused("preset_reserved")
        with self.storage.connect() as conn:
            if not conn.execute("DELETE FROM conversion_presets WHERE owner=? AND name=?", (OWNER, name)).rowcount:
                raise Refused("preset_unknown")
            conn.execute("UPDATE user_settings SET default_preset=NULL, updated=? WHERE owner=? AND default_preset=?",
                         (time.time(), OWNER, name))
        return self.presets()

    def set_default(self, name):
        """The default conversion preset of the user, or none with `name` None."""
        if name is not None:
            self.preset(name)
        with self.storage.reservation('control', 65536), self.storage.connect() as conn:
            conn.execute("INSERT INTO user_settings (owner, default_preset, updated) VALUES (?,?,?) ON CONFLICT(owner) "
                         "DO UPDATE SET default_preset=excluded.default_preset, updated=excluded.updated",
                         (OWNER, name, time.time()))
        return {"default": name}

    def preset_options(self, name, extra=None):
        """The job options of a conversion preset: its route, OCR, Datalab mode and, on the GPU route, its server mode
        and GPU preset (MFL023 B3), with only `lang` and `transit_confirmed` of the request beside them, and its name
        (`preset`). A preset the instance cannot use is refused with its code: the availability of the instance alone,
        as for the same options sent one by one (`job_options`, MFL023 B10, C-DB-NB2.2); what the worker last read
        stays for the list and the estimate, and the worker checks the rest before any arm."""
        extra = dict(extra or {})
        if set(extra) - {"lang", "transit_confirmed"}:
            raise Refused("invalid_setting")
        item = self.preset(name, worker=False)
        if not item["available"]:
            raise Refused(item["code"])
        options = {k: item["options"][k] for k in ("route", "ocr", "datalab_mode", "server_mode", "gpu_preset")
                   if item["options"][k] is not None}
        options.update(extra)
        options["preset"] = name
        return options

    def convert_preset(self, name, extra=None):
        """The job options of `mdf convert --preset NAME` (MFL051, D-4): the function of `mdf submit --preset` and of
        the API (`preset_options`), so the same preset is refused with the same code everywhere. One preset more is
        refused here: a preset on a rented GPU needs the session, the ledger and the caps of the worker, which a
        conversion outside the archive does not have (`preset_needs_archive`, whatever else the instance would say
        of it: the answer is to send the file to the archive)."""
        item = self.preset(name, worker=False)
        if item["options"].get("gpu_preset") is not None or item["options"].get("route") == routes.MARKER_SERVER:
            raise Refused("preset_needs_archive")
        return self.preset_options(name, extra)

    # --- the state of the routes (MFL051) ---------------------------------------------------------------------------

    def routes(self):
        """`GET /api/v1/routes`, `mdf routes`: the state of every route as the worker last computed it (`routestate`),
        read from its heartbeat and never computed here. `stale` when the heartbeat is older than the lease; without
        a heartbeat ever, no routes."""
        from . import routestate
        with self.storage.connect() as conn:
            row = conn.execute("SELECT heartbeat, readiness FROM worker WHERE id=1").fetchone()
        alive = bool(row is not None and time.time() - row["heartbeat"] < self.settings.lease_seconds)
        recorded = None
        if row is not None:
            try:
                value = json.loads(row["readiness"])
            except (TypeError, ValueError):
                value = None
            recorded = value.get("routes") if isinstance(value, dict) else None
        return routestate.answer(recorded, alive)

    def routes_refresh(self):
        """`POST /api/v1/routes/refresh`, `mdf routes --refresh`: ask the worker to check the routes at its next
        turn. The request is one row of `metadata` (no new table or column): the worker reads its time."""
        asked = time.time()
        with self.storage.connect() as conn:
            conn.execute("INSERT INTO metadata VALUES ('routes_refresh',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                         (repr(asked),))
        return {"requested_at": asked}

    def routes_refresh_asked(self):
        """The time of the last request of `routes_refresh`, 0.0 when there was none."""
        with self.storage.connect() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key='routes_refresh'").fetchone()
        try:
            return float(row["value"]) if row is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    # --- the GPU session (MFL023 B5) -------------------------------------------------------------------------------

    def worker_gpu(self):
        """The GPU part of the readiness the worker last recorded (closed values), or None."""
        with self.storage.connect() as conn:
            row = conn.execute("SELECT heartbeat, readiness FROM worker WHERE id=1").fetchone()
        if row is None or time.time() - row["heartbeat"] >= self.settings.lease_seconds:
            return None
        try:
            value = json.loads(row["readiness"])
        except (TypeError, ValueError):
            return None
        return value.get("gpu") if isinstance(value, dict) and isinstance(value.get("gpu"), dict) else None

    def gpu_session(self):
        """The GPU session of the user: the one not closed, else the last one; with, for a session that cannot be
        certified, orphan or stopped, the texts of the block and the commands that close it (the folder of the ledger of
        the instance appears only here, never in `Settings.public()`, a manifest or a ZIP)."""
        with self.storage.connect() as conn:
            row = conn.execute("SELECT * FROM gpu_sessions WHERE owner=? ORDER BY (state='closed'), updated DESC LIMIT 1",
                               (OWNER,)).fetchone()
            pending = [dict(r) for r in conn.execute("SELECT id, gpu_preset, kind, state, code, created FROM gpu_requests "
                                                     "WHERE owner=? ORDER BY created DESC LIMIT 5", (OWNER,))]
        gpu = self.worker_gpu() or {}
        if row is None:
            return {"session": None, "requests": pending, "ledgers": gpu.get("ledgers")}
        return {"session": session_view(dict(row), gpu.get("ledgers") or {}), "requests": pending,
                "ledgers": gpu.get("ledgers")}

    def gpu_arm(self, preset, transit_confirmed):
        """A request to arm `preset` ahead (plan r2 §4.5): only a preset that arms ahead (RunPod: the arm loads the
        image without RUNNING), usable by the instance, with the transit confirmed; the worker takes it between two
        jobs, and refuses it when it is older than 10 minutes."""
        from .gpu import presets as gpu_presets
        if not isinstance(preset, str):                  # MFL023 B10 (C-DB-NB4): before any other check
            raise Refused("invalid_setting")
        if transit_confirmed is not True:
            raise Refused("transit_not_confirmed")
        table = gpu_presets.load()
        if preset not in table:
            raise Refused("gpu_preset_unknown")
        state = gpu_availability(preset, self.settings)
        if not state["available"]:
            raise Refused(state["code"])
        if table[preset]["levers"]["arm_ahead_s"] is None:
            raise Refused("route_option_unsupported", "This GPU preset does not arm ahead: its provider gives nothing "
                                                      "before the first request")
        request = uuid.uuid4().hex
        with self.storage.reservation('control', 65536), self.storage.connect() as conn:
            now = time.time()
            conn.execute("INSERT INTO gpu_requests (id, owner, gpu_preset, kind, state, created, updated) VALUES "
                         "(?,?,?,?,'pending',?,?)", (request, OWNER, preset, "arm", now, now))
        return {"request": request, "gpu_preset": preset, "kind": "arm", "state": "pending"}

    def gpu_disarm(self, preset):
        """Disarm (plan r2 §4.5): in one transaction the conversions of `preset` in progress are cancelled at their next
        tick and those in the queue cancelled now ("session disarmed by the user"), then the request; the worker stops
        the session."""
        from .gpu import presets as gpu_presets
        if not isinstance(preset, str):                  # MFL023 B10 (C-DB-NB4): before any other check
            raise Refused("invalid_setting")
        if preset not in gpu_presets.load():
            raise Refused("gpu_preset_unknown")
        request = uuid.uuid4().hex
        with self.storage.reservation('control', 65536), self.storage.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            running = conn.execute("UPDATE runs SET cancel=1, updated=? WHERE state='running' AND "
                                   "json_extract(options,'$.gpu_preset')=?", (now, preset)).rowcount
            queued = conn.execute("UPDATE runs SET cancel=1, state='cancelled', error='Session disarmed by the user', "
                                  "updated=? WHERE state='queued' AND json_extract(options,'$.gpu_preset')=?",
                                  (now, preset)).rowcount
            conn.execute("INSERT INTO gpu_requests (id, owner, gpu_preset, kind, state, created, updated) VALUES "
                         "(?,?,?,?,'pending',?,?)", (request, OWNER, preset, "disarm", now, now))
        return {"request": request, "gpu_preset": preset, "kind": "disarm", "state": "pending",
                "cancelled": {"running": running, "queued": queued}}

    def gpu_estimate(self, preset):
        """What the user reads before sending a job on a GPU preset (plan r3 §4.8): the reserve `A` and the window `W`
        of a session now, the minimum reserve of a session (`minimum_reserve_cents`, never a charge), the cost of a
        start and of a page from the rate and the seconds of the preset, the measures with their date, hardware and
        source, the transit (provider, region, GPU), quality unverified; the values of the ledger come from the worker,
        with `open_session` when the open session of the preset hosts the job (MFL023 B10: `A` and `W` stay those of a
        new session)."""
        from decimal import Decimal, ROUND_CEILING
        from .gpu import presets as gpu_presets
        table = gpu_presets.load()
        if preset not in table:
            raise Refused("gpu_preset_unknown")
        p = table[preset]
        rate = Decimal(p["rate"]["usd_per_s"])
        cents = lambda s: str((rate * s * 100).quantize(Decimal("0.01"), rounding=ROUND_CEILING))  # noqa: E731
        worker = ((self.worker_gpu() or {}).get("presets") or {}).get(preset) or {}
        state = with_worker(gpu_availability(preset, self.settings), worker)
        group = p["gpu"]["group"]
        return {"gpu_preset": preset, "provider": p["provider"], **state,
                "region": p["data_center"] or "chosen by Modal", "gpu": group if isinstance(group, list) else [group],
                "a_cents": worker.get("a_cents"), "w_s": worker.get("w_s"),
                "minimum_reserve_cents": worker.get("minimum_reserve_cents"),
                "open_session": worker.get("open_session") is True,
                "start_cents": cents(p["seconds"]["cold_start_s"]), "page_cents": cents(p["seconds"]["page_seconds"]),
                "rate": p["rate"], "measures": p["measures"], "provisional": True, "quality": "unverified",
                "arm_ahead": p["levers"]["arm_ahead_s"] is not None, "notice": gpu_notice(preset, p),
                "bound_text": "A reserve and a measured limit are an upper bound, not a bill"}

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
