"""Loopback-only HTTP facade and inert static reader."""
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
import asyncio
import codecs
from decimal import Decimal
import hashlib
import json
import secrets
import sqlite3
import tempfile
import time
import uuid
from urllib.parse import unquote

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .engine import routes
from .engine.formats import FORMATS, MIME, extension_of
from .files import digest, open_file, IntegrityError, MetadataError
from .services import Refused, Service, ReadUnavailable, source_format, stored_manifest
from .storage import StorageError

CSP = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'"
# Shown by the Web UI before a conversion with Datalab, which needs the confirmation (MFL032 C4).
TRANSIT_NOTICE = ("With Datalab the document leaves this computer: it is uploaded to the Datalab API "
                  "(datalab.to) for the conversion, and its result is deleted there afterwards. Whether "
                  "Datalab keeps the document or uses it for training depends on the settings of the "
                  "Datalab account, not on this app. Send only documents you may share with Datalab.")


def worker_readiness(text):
    """{ready, message, converters} from the readiness the worker records: a JSON object since
    tranche C of MFL032 (availability per converter), a text before it."""
    try:
        value = json.loads(text) if text else None
    except ValueError:
        value = None
    if isinstance(value, dict) and isinstance(value.get('converters'), dict):
        return {'ready': value.get('ready') is True, 'message': str(value.get('message', '')),
                'converters': value['converters']}
    return {'ready': text == 'ready', 'message': text or '', 'converters': None}


def job_request(query, allowed=routes.JOB_OPTIONS):
    """The job options of a request from its query parameters; the routes validate them."""
    if set(query) - set(allowed):
        raise Refused('invalid_setting')
    options = {k: query[k] for k in ('route', 'ocr', 'lang', 'datalab_mode') if k in allowed and query.get(k)}
    if 'transit_confirmed' in query:
        if query['transit_confirmed'] not in ('true', 'false'):
            raise Refused('invalid_setting')
        options['transit_confirmed'] = query['transit_confirmed'] == 'true'
    return options


PRESET_BODY_BYTES = 16384


async def json_body(request, keys):
    """The JSON object of a small request body with exactly the fields `keys` (the conversion presets, MFL023 A5).
    The body is read piece by piece and refused as soon as it passes PRESET_BODY_BYTES, without receiving the
    pieces after that one (SOL-A-B1 of the review of A)."""
    parts, size = [], 0
    async for piece in request.stream():
        size += len(piece)
        if size > PRESET_BODY_BYTES:
            raise Refused('preset_invalid')
        parts.append(piece)
    data = b''.join(parts)
    try:
        value = json.loads(data)
    except ValueError:
        raise Refused('preset_invalid') from None
    if not isinstance(value, dict) or set(value) != set(keys):
        raise Refused('preset_invalid')
    return value


def create_app(settings):
    service = Service(settings)
    token = secrets.token_urlsafe(32)
    app = FastAPI(docs_url=None, redoc_url=None)
    app.state.service = service
    upload_lock = asyncio.Lock()
    allowed = {f'127.0.0.1:{settings.port}', f'localhost:{settings.port}'}

    @app.middleware('http')
    async def boundary(request, call_next):
        host = request.headers.get('host', '')
        if host not in allowed:
            return JSONResponse({'error': 'Host not allowed'}, status_code=403)
        if request.method not in {'GET', 'HEAD'}:
            if request.headers.get('origin') != 'http://' + host or not secrets.compare_digest(request.headers.get('x-mdf-csrf', ''), token):
                return JSONResponse({'error': 'Invalid origin or CSRF token'}, status_code=403)
        response = await call_next(request)
        response.headers.update({'Content-Security-Policy': CSP, 'X-Content-Type-Options': 'nosniff',
                                 'Referrer-Policy': 'no-referrer', 'Cache-Control': 'no-store',
                                 'Cross-Origin-Resource-Policy': 'same-origin'})
        return response

    @app.exception_handler(StorageError)
    async def storage_error(request, exc):
        body = {'error': str(exc)}
        if isinstance(exc, Refused):
            body['code'] = exc.code
        return JSONResponse(body, status_code=409)

    @app.exception_handler(ReadUnavailable)
    async def read_unavailable(request, exc):
        return JSONResponse({'error': str(exc), 'retryable': True}, status_code=503)

    @app.exception_handler(sqlite3.OperationalError)
    async def db_unavailable(request, exc):
        return JSONResponse({'error': 'Archive temporarily unavailable; retry',
                             'retryable': True}, status_code=503)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({'error': 'Invalid request'}, status_code=400)

    @app.exception_handler(OSError)
    async def io_error(request, exc):
        return JSONResponse({'error': 'File operation failed; archive preserved'}, status_code=409)

    @app.get('/health')
    def health():
        return {'alive': True}

    def worker_state():
        with service.storage.connect() as conn:
            row = conn.execute('SELECT * FROM worker WHERE id=1').fetchone()
        live = bool(row and time.time() - row['heartbeat'] < settings.lease_seconds)
        state = worker_readiness(row['readiness']) if live else None
        return {'ready': bool(state and state['ready']),
                'worker': state['message'] if live else 'Worker absent; archive available',
                'converters': state['converters'] if live else None}

    @app.get('/api/v1/readiness')
    def readiness():
        """The worker and, per converter, whether its route can run (MFL032 C2)."""
        return worker_state()

    @app.get('/api/v1/session')
    def session():
        """The formats of the matrix with their routes, the availability of each converter as the
        worker last checked it, and Datalab: configured when the worker has its key and the
        spending cap of MDF_DATALAB_CAP_USD is above 0."""
        state = worker_state()
        converters = state['converters'] or {}
        cap = Decimal(settings.datalab_cap_usd)
        available = (converters.get(routes.DATALAB) or {}).get('available') is True
        return {'csrf': token, 'upload_mib': settings.upload_mib,
                'formats': [{'format': fmt, 'family': family, 'extensions': list(exts), **routes.routes_for(fmt)}
                            for fmt, (family, exts) in FORMATS.items()],
                'converters': state['converters'], 'remote': sorted(routes.REMOTE),
                'datalab': {'configured': available and cap > 0, 'cap_usd': settings.datalab_cap_usd,
                            'modes': list(routes.DATALAB_MODES), 'notice': TRANSIT_NOTICE}}

    @app.get('/api/v1/runs')
    def runs():
        return service.list()

    @app.get('/api/v1/runs/{run_id}')
    def get(run_id: str):
        return service.get(run_id)

    def preset_request(query):
        """`?preset=NAME`: the options of a conversion preset, with only `lang` and `transit_confirmed` beside it."""
        extra = job_request({k: v for k, v in query.items() if k != 'preset'}, allowed=('lang', 'transit_confirmed'))
        return service.preset_options(query['preset'], extra)

    @app.post('/api/v1/runs', status_code=202)
    async def submit(request: Request):
        name = unquote(request.headers.get('x-filename', 'document.pdf'), errors='strict')
        key = request.headers.get('idempotency-key')
        query = request.query_params
        options = await run_in_threadpool(preset_request, query) if 'preset' in query else job_request(query)
        # Hold only this upload's lease while receiving, never an admission/DB lock.
        context = service.upload(name, options, key=key)
        receiver = await run_in_threadpool(context.__enter__)
        try:
            async for chunk in request.stream():
                await run_in_threadpool(receiver.write, chunk)
            return await run_in_threadpool(receiver.finish)
        finally:
            await run_in_threadpool(context.__exit__, None, None, None)

    @app.post('/api/v1/runs/{run_id}/retry', status_code=202)
    def retry(run_id: str, request: Request):
        # the route and the options of the run; a Datalab run confirms its transit again
        return service.retry(run_id, job_request(request.query_params, allowed=('transit_confirmed',)))

    @app.get('/api/v1/presets')
    def presets():
        """The conversion presets of the instance and of the user, the default of the user and the GPU presets."""
        return service.presets()

    @app.post('/api/v1/presets', status_code=201)
    async def add_preset(request: Request):
        body = await json_body(request, ('name', 'options'))
        return await run_in_threadpool(service.add_preset, body['name'], body['options'])

    @app.put('/api/v1/presets/{name}')
    async def replace_preset(name: str, request: Request):
        body = await json_body(request, ('options',))
        return await run_in_threadpool(service.replace_preset, name, body['options'])

    @app.delete('/api/v1/presets/{name}')
    def remove_preset(name: str):
        return service.remove_preset(name)

    @app.put('/api/v1/settings/default-preset')
    async def default_preset(request: Request):
        body = await json_body(request, ('name',))
        return await run_in_threadpool(service.set_default, body['name'])

    @app.post('/api/v1/runs/{run_id}/cancel')
    def cancel(run_id: str):
        return service.cancel(run_id)

    def artifact(run_id, artifact_id):
        row = service.storage.get(run_id)
        if not row['manifest'] or not row['result_path']:
            raise StorageError('Result not available')
        try:
            from .files import read_json
            manifest = stored_manifest(row['manifest'], row)
            if read_json(service.storage.root, row['result_path'] + '/manifest.json') != manifest:
                raise IntegrityError('Manifest mismatch')
        except MetadataError as exc:
            service.read_failure(run_id, IntegrityError('Malformed stored manifest'))
        except (OSError, ValueError) as exc:
            service.read_failure(run_id, exc)
        selected = next((a for a in manifest['artifacts'] if a['id'] == artifact_id), None)
        if selected is None:
            raise StorageError('Artifact not found')
        path = row['result_path'] + '/original/' + selected['path']
        try:
            with open_file(service.storage.root, path) as stream:
                actual = digest(stream)
            if any(actual[k] != selected[k] for k in ('bytes', 'sha256')):
                raise IntegrityError('Identity mismatch')
        except (OSError, ValueError) as exc:
            service.read_failure(run_id, exc)
        return selected, path

    @app.get('/api/v1/runs/{run_id}/text')
    def text(run_id: str, offset: int = 0):
        job = service.get(run_id)
        if not job['manifest']:
            raise StorageError('Text not available yet')
        raw = next(a for a in job['manifest']['artifacts'] if a['path'] == 'document.md')
        selected, path = artifact(run_id, raw['id'])
        if offset < 0 or offset > selected['bytes']:
            raise HTTPException(400, 'Invalid text offset')
        try:
            with open_file(service.storage.root, path) as stream:
                # Validate the document independently of the request offset.
                # Invalid UTF-8 must not be disguised as an alignment error.
                decoder = codecs.getincrementaldecoder('utf-8')()
                try:
                    while chunk := stream.read(65536):
                        decoder.decode(chunk)
                    decoder.decode(b'', final=True)
                except UnicodeDecodeError as exc:
                    raise IntegrityError('Unreadable stored UTF-8') from exc
                stream.seek(offset)
                data = stream.read(256 * 1024)
                if data and data[0] & 0xc0 == 0x80:
                    raise HTTPException(400, 'Invalid text offset')
                for _ in range(4):
                    try:
                        content = data.decode('utf-8')
                        break
                    except UnicodeDecodeError as exc:
                        if exc.end != len(data):
                            raise IntegrityError('Unreadable stored UTF-8') from exc
                        data += stream.read(1)
                else:
                    raise IntegrityError('Unreadable stored UTF-8')
                next_offset = stream.tell()
        except (OSError, ValueError) as exc:
            service.read_failure(run_id, exc)
        return {'text': content, 'offset': offset, 'next_offset': next_offset, 'total_bytes': selected['bytes'], 'complete': next_offset == selected['bytes']}

    @app.get('/api/v1/runs/{run_id}/artifacts/{artifact_id}')
    def asset(run_id: str, artifact_id: str):
        selected, path = artifact(run_id, artifact_id)
        if not selected['gallery'] or selected.get('mime') not in {'image/png', 'image/jpeg'}:
            raise StorageError('Artifact not enabled for the gallery')
        try:
            with open_file(service.storage.root, path) as stream:
                data = stream.read(settings.output_mib * 1048576 + 1)
        except (OSError, ValueError) as exc:
            service.read_failure(run_id, exc)
        if len(data) != selected['bytes'] or hashlib.sha256(data).hexdigest() != selected['sha256']:
            service.invalidate(run_id, 'Artifact altered while read')
            raise StorageError('Artifact altered')
        return Response(data, media_type=selected['mime'])

    @app.get('/api/v1/runs/{run_id}/source')
    def source(run_id: str):
        row = service.storage.get(run_id)
        try:
            with open_file(service.storage.root, row['source_path']) as stream:
                data = stream.read(settings.upload_mib * 1048576 + 1)
        except (OSError, ValueError) as exc:
            service.read_failure(run_id, exc)
        if len(data) != row['source_bytes'] or hashlib.sha256(data).hexdigest() != row['source_sha256']:
            service.invalidate(run_id, 'Source altered')
            raise StorageError('Source altered')
        name = 'document.' + extension_of(row['source_path'])
        return Response(data, media_type=MIME[source_format(row['source_path'])],
                        headers={'Content-Disposition': f'attachment; filename="{name}"'})

    @app.get('/api/v1/runs/{run_id}/bundle')
    def bundle(run_id: str):
        path = service.bundle(run_id)
        def stream():
            with open_file(service.storage.root, str(path.relative_to(service.storage.root))) as source:
                while chunk := source.read(65536):
                    yield chunk
        return StreamingResponse(stream(), media_type='application/zip', headers={'Content-Disposition': f'attachment; filename="{run_id}.zip"'})

    @app.get('/')
    def index():
        return Response(resources.files('markdown_for_llms').joinpath('web/index.html').read_bytes(), media_type='text/html')

    @app.get('/static/{name}')
    def static(name: str):
        types = {'app.js': 'text/javascript', 'style.css': 'text/css'}
        if name not in types:
            return Response(status_code=404)
        return Response(resources.files('markdown_for_llms').joinpath('web/' + name).read_bytes(), media_type=types[name])

    return app
