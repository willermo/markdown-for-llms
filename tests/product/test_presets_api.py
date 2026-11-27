"""The conversion presets through the API (MFL023 A5, D7): those of the instance, provisional, and the user's own with the
same closed fields and a unique name; the default of the user; `POST /api/v1/runs?preset=NAME` with only `lang` and
`transit_confirmed` beside it, its name recorded in the job; the GPU presets listed with the checks of the instance
(since MFL023 B3: `gpu_preset_not_enabled`, `server_cap` with the caps at 0); CSRF on every change; both catalogs in the
wheel."""
import asyncio
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient
from starlette.requests import Request

from markdown_for_llms.api import PRESET_BODY_BYTES, create_app, json_body
from markdown_for_llms.services import Refused, builtin_presets
from markdown_for_llms.settings import Settings

PDF = b'%PDF-1.4\nsynthetic contract input; not a valid conversion fixture\n'
MINE = {"route": "marker", "ocr": "force"}


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(data_root=self.temp.name + '/data', model_root=self.temp.name + '/models',
                                 gpu_presets="modal-base")
        self.client = TestClient(create_app(self.settings), base_url=f'http://127.0.0.1:{self.settings.port}')
        self.csrf = self.client.get('/api/v1/session').json()['csrf']
        self.headers = {'origin': f'http://127.0.0.1:{self.settings.port}', 'x-mdf-csrf': self.csrf}

    def call(self, method, url, body=None, **kw):
        content = json.dumps(body) if body is not None else kw.pop('content', None)
        return self.client.request(method, url, content=content, headers=dict(self.headers, **kw.pop('headers', {})), **kw)

    def run_options(self, run_id):
        with sqlite3.connect(Path(self.settings.data_root) / 'archive.sqlite3') as conn:
            value = conn.execute('SELECT options FROM runs WHERE id=?', (run_id,)).fetchone()[0]
        conn.close()
        return json.loads(value)


class ListTests(Base):
    def test_the_presets_of_the_instance_and_the_gpu_presets(self):
        body = self.client.get('/api/v1/presets').json()
        builtin = {p['name']: p for p in body['presets'] if p['builtin']}
        self.assertEqual(sorted(builtin), ['batch', 'datalab', 'local-fast', 'local-gpu', 'quality-now', 'remote-gpu'])
        self.assertTrue(all(p['provisional'] for p in builtin.values()))
        self.assertEqual({n: (p['available'], p['code']) for n, p in builtin.items()}, {
            'local-fast': (True, None), 'datalab': (True, None),
            'quality-now': (False, 'server_cap'), 'batch': (False, 'gpu_preset_not_enabled'),
            'local-gpu': (False, 'gpu_preset_not_enabled'), 'remote-gpu': (False, 'gpu_preset_not_enabled')})
        self.assertEqual((builtin['quality-now']['options']['gpu_preset'], builtin['batch']['options']['gpu_preset']),
                         ('modal-base', 'runpod-serverless-volume-dc'))
        self.assertIsNone(body['default'])
        gpus = {g['name']: g for g in body['gpu_presets']}
        self.assertEqual({n: (g['enabled'], g['available'], g['code']) for n, g in gpus.items()}, {
            'modal-base': (True, False, 'server_cap'),
            'runpod-serverless-volume-dc': (False, False, 'gpu_preset_not_enabled'),
            'runpod-serverless-volume-eu-se-1': (False, False, 'gpu_preset_superseded'),
            'gpu-local': (False, False, 'gpu_preset_not_enabled'), 'gpu-remote': (False, False, 'gpu_preset_not_enabled')})
        self.assertEqual(gpus['modal-base']['measures'][0]['value'], '164-172 s')
        self.assertNotIn(self.temp.name, json.dumps(body))


class ChangeTests(Base):
    def test_a_preset_of_the_user(self):
        made = self.call('POST', '/api/v1/presets', {'name': 'mine', 'options': MINE})
        self.assertEqual(made.status_code, 201, made.text)
        self.assertEqual(made.json()['options'], {'route': 'marker', 'server_mode': None, 'gpu_preset': None,
                                                  'ocr': 'force', 'datalab_mode': None})
        self.assertFalse(made.json()['builtin'])
        again = self.call('POST', '/api/v1/presets', {'name': 'mine', 'options': MINE})
        self.assertEqual((again.status_code, again.json()['code']), (409, 'preset_exists'))
        changed = self.call('PUT', '/api/v1/presets/mine', {'options': {'route': 'datalab', 'datalab_mode': 'fast'}})
        self.assertEqual(changed.json()['options']['datalab_mode'], 'fast')
        self.assertEqual(self.call('PUT', '/api/v1/settings/default-preset', {'name': 'mine'}).json(), {'default': 'mine'})
        self.assertEqual(self.client.get('/api/v1/presets').json()['default'], 'mine')
        removed = self.call('DELETE', '/api/v1/presets/mine')
        self.assertEqual(removed.status_code, 200)
        self.assertIsNone(removed.json()['default'])                  # the default goes with its preset
        gone = self.call('DELETE', '/api/v1/presets/mine')
        self.assertEqual((gone.status_code, gone.json()['code']), (409, 'preset_unknown'))
        self.assertEqual(self.call('PUT', '/api/v1/settings/default-preset', {'name': 'batch'}).json(), {'default': 'batch'})
        self.assertEqual(self.call('PUT', '/api/v1/settings/default-preset', {'name': None}).json(), {'default': None})

    def test_refusals(self):
        cases = [
            ('POST', '/api/v1/presets', {'name': 'local-fast', 'options': MINE}, 'preset_reserved'),
            ('PUT', '/api/v1/presets/batch', {'options': MINE}, 'preset_reserved'),
            ('DELETE', '/api/v1/presets/datalab', None, 'preset_reserved'),
            ('PUT', '/api/v1/presets/nope', {'options': MINE}, 'preset_unknown'),
            ('PUT', '/api/v1/settings/default-preset', {'name': 'nope'}, 'preset_unknown'),
            ('POST', '/api/v1/presets', {'name': 'Bad Name', 'options': MINE}, 'preset_invalid'),
            ('POST', '/api/v1/presets', {'name': 'x', 'options': MINE, 'extra': 1}, 'preset_invalid'),
            ('POST', '/api/v1/presets', {'name': 'x', 'options': dict(MINE, lang='it')}, 'preset_invalid'),
            ('POST', '/api/v1/presets', {'name': 'x', 'options': {'route': 'nope'}}, 'preset_invalid'),
            ('POST', '/api/v1/presets', {'name': 'x', 'options': {'route': 'marker', 'gpu_preset': 'modal-base'}},
             'preset_invalid'),
            ('POST', '/api/v1/presets', {'name': 'x', 'options': {'route': 'marker-server', 'server_mode': 'balanced'}},
             'preset_invalid'),
            ('POST', '/api/v1/presets', {'name': 'x', 'options': {'route': 'marker', 'datalab_mode': 'fast'}},
             'preset_invalid'),
            ('POST', '/api/v1/presets', {'name': 'x', 'options': {'route': 'marker-server', 'server_mode': 'balanced',
                                                                  'gpu_preset': 'runpod-serverless-volume-eu-se-1'}},
             'gpu_preset_superseded'),
            ('POST', '/api/v1/presets', {'name': 'x', 'options': {'route': 'marker-server', 'server_mode': 'fast',
                                                                  'gpu_preset': 'modal-base'}}, 'preset_invalid'),
            ('POST', '/api/v1/presets', {'name': 'x', 'options': {'route': 'marker-server', 'server_mode': 'balanced',
                                                                  'gpu_preset': 'nope'}}, 'gpu_preset_unknown'),
        ]
        for method, url, body, code in cases:
            with self.subTest(url=url, body=body):
                answer = self.call(method, url, body)
                self.assertEqual((answer.status_code, answer.json().get('code')), (409, code), answer.text)
        self.assertEqual(self.call('POST', '/api/v1/presets', content='{not json').json()['code'], 'preset_invalid')
        self.assertEqual(self.call('POST', '/api/v1/presets', content='x' * 20000).json()['code'], 'preset_invalid')
        self.assertEqual([p['name'] for p in self.client.get('/api/v1/presets').json()['presets'] if not p['builtin']], [])

    def test_a_large_body_is_refused_before_it_is_read(self):
        """SOL-A-B1 of the review of A: a body sent in pieces is refused (`preset_invalid`) as soon as its bytes pass
        PRESET_BODY_BYTES, through the whole app, and the pieces after that one are never received."""
        piece, pieces = 4096, 10
        first_over = PRESET_BODY_BYTES // piece + 1
        received = []

        async def receive():
            received.append(len(received))
            if len(received) > pieces:
                return {'type': 'http.disconnect'}
            return {'type': 'http.request', 'body': b'x' * piece, 'more_body': len(received) < pieces}
        sent = []

        async def send(message):
            sent.append(message)
        host = '127.0.0.1:%d' % self.settings.port
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
                 'path': '/api/v1/presets', 'raw_path': b'/api/v1/presets', 'query_string': b'', 'root_path': '',
                 'client': ('127.0.0.1', 1), 'server': ('127.0.0.1', self.settings.port),
                 'headers': [(b'host', host.encode()), (b'origin', b'http://' + host.encode()),
                             (b'x-mdf-csrf', self.csrf.encode()), (b'content-type', b'application/json')]}
        asyncio.run(self.client.app(scope, receive, send))
        body = b''.join(m.get('body', b'') for m in sent if m['type'] == 'http.response.body')
        self.assertEqual((sent[0]['status'], json.loads(body)['code']), (409, 'preset_invalid'))
        self.assertEqual(len(received), first_over)

    def test_json_body_reads_pieces_within_the_limit(self):
        async def read(chunks):
            queue = list(chunks)

            async def receive():
                return {'type': 'http.request', 'body': queue.pop(0), 'more_body': bool(queue)}
            request = Request({'type': 'http', 'method': 'POST', 'path': '/', 'headers': []}, receive)
            return await json_body(request, ('name', 'options'))
        whole = json.dumps({'name': 'mine', 'options': MINE}).encode()
        self.assertEqual(asyncio.run(read([whole[i:i + 3] for i in range(0, len(whole), 3)])),
                         {'name': 'mine', 'options': MINE})
        exact = whole + b' ' * (PRESET_BODY_BYTES - len(whole))
        self.assertEqual(asyncio.run(read([exact[:100], exact[100:]]))['name'], 'mine')
        for chunks in ([exact, b' '], [b'{not', b' json'], [b''], [b'[1]']):
            with self.subTest(chunks=chunks[-1]):
                with self.assertRaises(Refused) as ctx:
                    asyncio.run(read(chunks))
                self.assertEqual(ctx.exception.code, 'preset_invalid')

    def test_every_change_needs_the_csrf_token(self):
        for method, url, body in (('POST', '/api/v1/presets', {'name': 'mine', 'options': MINE}),
                                  ('PUT', '/api/v1/presets/mine', {'options': MINE}),
                                  ('DELETE', '/api/v1/presets/mine', None),
                                  ('PUT', '/api/v1/settings/default-preset', {'name': 'batch'})):
            with self.subTest(method=method, url=url):
                answer = self.client.request(method, url, content=json.dumps(body) if body else None,
                                             headers={'origin': self.headers['origin']})
                self.assertEqual(answer.status_code, 403)


class SubmitTests(Base):
    def submit(self, query):
        return self.call('POST', '/api/v1/runs?' + query, content=PDF, headers={'x-filename': 'a.pdf'})

    def test_a_job_with_a_preset(self):
        answer = self.submit('preset=local-fast&lang=it')
        self.assertEqual(answer.status_code, 202, answer.text)
        self.assertEqual(self.run_options(answer.json()['id']), {'route': None, 'ocr': 'auto', 'lang': 'it',
                                                                 'datalab_mode': None, 'transit_confirmed': False,
                                                                 'preset': 'local-fast'})
        self.assertEqual(answer.json()['options']['preset'], 'local-fast')
        self.call('POST', '/api/v1/presets', {'name': 'mine', 'options': MINE})
        mine = self.submit('preset=mine')
        self.assertEqual(self.run_options(mine.json()['id'])['ocr'], 'force')
        dl = self.submit('preset=datalab&transit_confirmed=true')
        self.assertEqual(dl.status_code, 202, dl.text)
        self.assertEqual((self.run_options(dl.json()['id'])['route'], self.run_options(dl.json()['id'])['datalab_mode']),
                         ('datalab', 'balanced'))

    def test_a_job_without_a_preset_stores_no_key_more(self):
        answer = self.submit('route=marker')
        self.assertEqual(sorted(self.run_options(answer.json()['id'])),
                         ['datalab_mode', 'lang', 'ocr', 'route', 'transit_confirmed'])

    def test_refused_jobs(self):
        for query, code in (('preset=quality-now', 'server_cap'), ('preset=batch', 'gpu_preset_not_enabled'),
                            ('preset=nope', 'preset_unknown'), ('preset=local-fast&route=marker', 'invalid_setting'),
                            ('preset=local-fast&ocr=force', 'invalid_setting'),
                            ('preset=local-fast&datalab_mode=fast', 'invalid_setting'),
                            ('preset=datalab', 'transit_not_confirmed'), ('preset=local-fast&transit_confirmed=yes',
                                                                          'invalid_setting'),
                            ('preset=', 'preset_unknown')):
            with self.subTest(query=query):
                answer = self.submit(query)
                self.assertEqual((answer.status_code, answer.json().get('code')), (409, code), answer.text)
        self.assertEqual(self.client.get('/api/v1/runs').json(), [])


class TheWheelTests(unittest.TestCase):
    def test_both_catalogs_load_from_the_installed_package(self):
        """`package-data` (SOL-R6 = C-NB1 of the plan r1): an isolated process started outside the checkout loads the
        conversion presets and the GPU presets from the installed package. Green once the full check has installed
        the canonical wheel."""
        code = ("import json; from markdown_for_llms import services; from markdown_for_llms.gpu import presets; "
                "print(json.dumps([str(services.CONVERSION_PRESETS), sorted(services.builtin_presets()), "
                "str(presets.PATH), sorted(presets.load())]))")
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, cwd=tmp,
                                 timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        conv_path, conv, gpu_path, gpu = json.loads(out.stdout)
        self.assertIn('site-packages', Path(conv_path).parts)
        self.assertIn('site-packages', Path(gpu_path).parts)
        self.assertEqual(conv, sorted(builtin_presets()))
        self.assertEqual(len(gpu), 5)                  # MFL051 B: gpu-local and gpu-remote


if __name__ == '__main__':
    unittest.main()
