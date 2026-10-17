"""Loopback-only HTTP facade and inert static reader."""
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
import asyncio
import codecs
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

from .files import digest, open_file, IntegrityError, MetadataError
from .services import Service, ReadUnavailable, stored_manifest
from .storage import StorageError

CSP = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'"


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
            return JSONResponse({'error': 'Host non consentito'}, status_code=403)
        if request.method not in {'GET', 'HEAD'}:
            if request.headers.get('origin') != 'http://' + host or not secrets.compare_digest(request.headers.get('x-mdf-csrf', ''), token):
                return JSONResponse({'error': 'Origine o token CSRF non valido'}, status_code=403)
        response = await call_next(request)
        response.headers.update({'Content-Security-Policy': CSP, 'X-Content-Type-Options': 'nosniff',
                                 'Referrer-Policy': 'no-referrer', 'Cache-Control': 'no-store',
                                 'Cross-Origin-Resource-Policy': 'same-origin'})
        return response

    @app.exception_handler(StorageError)
    async def storage_error(request, exc):
        return JSONResponse({'error': str(exc)}, status_code=409)

    @app.exception_handler(ReadUnavailable)
    async def read_unavailable(request, exc):
        return JSONResponse({'error': str(exc), 'retryable': True}, status_code=503)

    @app.exception_handler(sqlite3.OperationalError)
    async def db_unavailable(request, exc):
        return JSONResponse({'error': 'Archivio temporaneamente indisponibile; riprovare',
                             'retryable': True}, status_code=503)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({'error': 'Richiesta non valida'}, status_code=400)

    @app.exception_handler(OSError)
    async def io_error(request, exc):
        return JSONResponse({'error': 'Operazione su file non riuscita; archivio conservato'}, status_code=409)

    @app.get('/health')
    def health():
        return {'alive': True}

    @app.get('/api/v1/readiness')
    def readiness():
        with service.storage.connect() as conn:
            row = conn.execute('SELECT * FROM worker WHERE id=1').fetchone()
        live = bool(row and time.time() - row['heartbeat'] < settings.lease_seconds)
        return {'ready': bool(live and row['readiness'] == 'ready'),
                'worker': row['readiness'] if live else 'Worker assente; archivio consultabile'}

    @app.get('/api/v1/session')
    def session():
        return {'csrf': token, 'upload_mib': settings.upload_mib}

    @app.get('/api/v1/runs')
    def runs():
        return service.list()

    @app.get('/api/v1/runs/{run_id}')
    def get(run_id: str):
        return service.get(run_id)

    @app.post('/api/v1/runs', status_code=202)
    async def submit(request: Request):
        name = unquote(request.headers.get('x-filename', 'document.pdf'), errors='strict')
        key = request.headers.get('idempotency-key')
        # Hold only this upload's lease while receiving, never an admission/DB lock.
        context = service.upload(name, key=key)
        receiver = await run_in_threadpool(context.__enter__)
        try:
            async for chunk in request.stream():
                await run_in_threadpool(receiver.write, chunk)
            return await run_in_threadpool(receiver.finish)
        finally:
            await run_in_threadpool(context.__exit__, None, None, None)

    @app.post('/api/v1/runs/{run_id}/retry', status_code=202)
    def retry(run_id: str):
        return service.retry(run_id)

    @app.post('/api/v1/runs/{run_id}/cancel')
    def cancel(run_id: str):
        return service.cancel(run_id)

    def artifact(run_id, artifact_id):
        row = service.storage.get(run_id)
        if not row['manifest'] or not row['result_path']:
            raise StorageError('Risultato non disponibile')
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
            raise StorageError('Artefatto non trovato')
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
            raise StorageError('Testo non ancora disponibile')
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
            raise StorageError('Artefatto non abilitato per la galleria')
        try:
            with open_file(service.storage.root, path) as stream:
                data = stream.read(settings.output_mib * 1048576 + 1)
        except (OSError, ValueError) as exc:
            service.read_failure(run_id, exc)
        if len(data) != selected['bytes'] or hashlib.sha256(data).hexdigest() != selected['sha256']:
            service.invalidate(run_id, 'Artefatto alterato durante la lettura')
            raise StorageError('Artefatto alterato')
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
            service.invalidate(run_id, 'Sorgente alterato')
            raise StorageError('Sorgente alterato')
        return Response(data, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename="document.pdf"'})

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
