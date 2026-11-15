"""The conversion matrix through the product (MFL032 C2): admission by extension and signature on
the whole staged file (K3, PC1), the job options validated by the routes of the engine, sources,
ZIP and /source with the extension and media type of their format, the upload reservation of
any format collected by the recovery (K4), the API session and readiness. No ML import."""
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
import uuid
import zipfile

from fastapi.testclient import TestClient

from markdown_for_llms.api import create_app
from markdown_for_llms.capacity import recover, upload_source
from markdown_for_llms.engine.formats import FORMATS
from markdown_for_llms.files import identify, write, write_json
from markdown_for_llms.services import Refused, Service, stored_options
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import StorageError
from markdown_for_llms.worker import Worker
from unittest.mock import patch


def zip_bytes(members, stored_first=None):
    body = io.BytesIO()
    with zipfile.ZipFile(body, 'w') as z:
        if stored_first is not None:
            z.writestr(zipfile.ZipInfo('padding.bin'), stored_first)        # stored, before the members
        for name, data in members.items():
            z.writestr(name, data)
    return body.getvalue()


def ooxml(main):
    return zip_bytes({'[Content_Types].xml': '<Types/>', main: '<x/>'})


def odf(mimetype):
    return zip_bytes({'mimetype': mimetype, 'content.xml': '<x/>'})


SAMPLES = {
    'pdf': ('a.pdf', b'%PDF-1.4\nsynthetic\n'),
    'png': ('a.png', b'\x89PNG\r\n\x1a\n' + b'\0' * 16),
    'jpeg': ('a.jpg', b'\xff\xd8\xff\xe0' + b'\0' * 16),
    'tiff': ('a.tif', b'II*\x00' + b'\0' * 16),
    'djvu': ('a.djvu', b'AT&TFORM\x00\x00\x00\x10DJVU' + b'\0' * 8),
    'docx': ('a.docx', ooxml('word/document.xml')),
    'xlsx': ('a.xlsx', ooxml('xl/workbook.xml')),
    'pptx': ('a.pptx', ooxml('ppt/presentation.xml')),
    'odt': ('a.odt', odf('application/vnd.oasis.opendocument.text')),
    'ods': ('a.ods', odf('application/vnd.oasis.opendocument.spreadsheet')),
    'epub': ('a.epub', odf('application/epub+zip')),
    'doc': ('a.doc', b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1' + b'\0' * 16),
    'rtf': ('a.rtf', b'{\\rtf1 synthetic}'),
    'html': ('a.html', b'<!doctype html><title>t</title>'),
    'txt': ('a.txt', 'Testo semplice, già UTF-8.\n'.encode()),
}


class FormatsThroughTheProductTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(data_root=self.temp.name + '/data', model_root=self.temp.name + '/models')
        self.service = Service(self.settings)
        self.addCleanup(self.service.storage.close)

    def refused(self, code, *args, **kwargs):
        with self.assertRaises(Refused) as raised:
            self.service.submit(*args, **kwargs)
        self.assertEqual(raised.exception.code, code)
        return raised.exception

    def test_every_format_of_the_matrix_is_admitted(self):                          # 12.1 through the product
        self.assertEqual(set(SAMPLES), set(FORMATS))
        for fmt, (name, data) in SAMPLES.items():
            with self.subTest(fmt=fmt):
                job = self.service.submit(io.BytesIO(data), name)
                row = self.service.storage.get(job['id'])
                self.assertEqual(row['source_path'], 'sources/%s.%s' % (row['source_path'][8:40], name.rsplit('.', 1)[1]))
                self.assertEqual((job['format'], job['state']), (fmt, 'queued'))
                self.assertEqual(job['options'], {'route': None, 'ocr': 'auto', 'lang': None, 'datalab_mode': None,
                                                  'transit_confirmed': False})

    def test_a_zip_package_is_recognized_from_its_central_directory(self):          # K3
        data = zip_bytes({'[Content_Types].xml': '<Types/>', 'word/document.xml': '<x/>'}, stored_first=os.urandom(12000))
        self.assertGreater(data.index(b'word/document.xml'), 8192)
        self.assertEqual(self.service.submit(io.BytesIO(data), 'long.docx')['format'], 'docx')
        self.refused('format_mismatch', io.BytesIO(b'%PDF-1.4\n'), 'fake.docx')
        self.refused('format_unsupported', io.BytesIO(b'MZ'), 'tool.exe')
        self.refused('format_unsupported', io.BytesIO(b'x'), 'no-extension')
        self.refused('input_unreadable', io.BytesIO(b'PK\x03\x04' + b'\0' * 64), 'damaged.docx')
        self.assertEqual(len(self.service.list()), 1)
        self.assertEqual(list((self.service.storage.root / 'staging').iterdir()), [])
        self.assertEqual(len(list((self.service.storage.root / 'sources').iterdir())), 1)

    def test_replay_compares_the_extension_of_the_registered_source(self):            # PC1
        html = b'<!doctype html><p>same bytes</p>'
        first = self.service.submit(io.BytesIO(html), 'page.html', key='k')
        self.assertEqual(self.service.submit(io.BytesIO(html), 'other-name.html', key='k')['id'], first['id'])
        for name in ('page.htm', 'page.txt'):
            with self.subTest(name=name), self.assertRaisesRegex(StorageError, 'Idempotency key conflicts'):
                self.service.submit(io.BytesIO(html), name, key='k')
        with self.assertRaisesRegex(StorageError, 'Idempotency key conflicts'):
            self.service.submit(io.BytesIO(html), 'page.html', {'route': 'marker'}, key='k')
        self.assertEqual(len(self.service.list()), 1)

    def test_job_options_are_validated_by_the_routes(self):
        pdf, docx, txt = SAMPLES['pdf'][1], SAMPLES['docx'][1], SAMPLES['txt'][1]
        cases = [('route_invalid', txt, 'a.txt', {'route': 'marker'}),
                 ('route_invalid', docx, 'a.docx', {'route': 'nonsense'}),
                 ('route_option_unsupported', docx, 'a.docx', {'ocr': 'force'}),
                 ('route_option_unsupported', pdf, 'a.pdf', {'datalab_mode': 'fast'}),
                 ('route_option_unsupported', pdf, 'a.pdf', {'transit_confirmed': True}),
                 ('route_option_unsupported', pdf, 'a.pdf', {'route': 'datalab', 'transit_confirmed': True, 'ocr': 'force'}),
                 ('transit_not_confirmed', pdf, 'a.pdf', {'route': 'datalab'}),
                 ('invalid_setting', pdf, 'a.pdf', {'lang': 'it; rm -rf'}),
                 ('invalid_setting', pdf, 'a.pdf', {'force_ocr': True})]
        for code, data, name, options in cases:
            with self.subTest(options=options):
                self.refused(code, io.BytesIO(data), name, options)
        self.assertEqual(self.service.list(), [])
        job = self.service.submit(io.BytesIO(pdf), 'a.pdf', {'route': 'datalab', 'transit_confirmed': True,
                                                             'datalab_mode': 'fast', 'lang': 'it'})
        self.assertEqual(job['options'], {'route': 'datalab', 'ocr': 'auto', 'lang': 'it', 'datalab_mode': 'fast',
                                          'transit_confirmed': True})
        self.assertEqual(self.service.submit(io.BytesIO(docx), 'a.docx', {'route': 'marker'})['options']['route'], 'marker')

    def test_a_retry_confirms_a_transit_again_and_maps_the_options_of_schema_2(self):
        pdf = SAMPLES['pdf'][1]
        job = self.service.submit(io.BytesIO(pdf), 'a.pdf', {'route': 'datalab', 'transit_confirmed': True})
        self.service.cancel(job['id'])
        with self.assertRaises(Refused) as raised:
            self.service.retry(job['id'])
        self.assertEqual(raised.exception.code, 'transit_not_confirmed')
        self.assertTrue(self.service.retry(job['id'], {'transit_confirmed': True})['options']['transit_confirmed'])
        # a run of schema 2: force_ocr becomes ocr force, strip_existing_ocr has no equivalent
        old = self.service.submit(io.BytesIO(pdf), 'b.pdf')
        v2 = dict(use_llm=False, force_ocr=True, strip_existing_ocr=True, disable_image_extraction=False,
                  paginate_output=True, pdftext_workers=1, layout_batch_size=1, recognition_batch_size=1,
                  detection_batch_size=1, table_rec_batch_size=1, ocr_error_batch_size=1,
                  lowres_image_dpi=96, highres_image_dpi=192)
        with self.service.storage.connect() as conn:
            conn.execute("UPDATE runs SET options=?, state='failed' WHERE id=?", (json.dumps(v2), old['id']))
        self.assertEqual(self.service.get(old['id'])['options']['ocr'], 'force')
        retry = self.service.retry(old['id'])
        self.assertEqual(retry['options'], {'route': None, 'ocr': 'force', 'lang': None, 'datalab_mode': None,
                                            'transit_confirmed': False})
        self.assertEqual(stored_options(json.dumps(v2))['ocr'], 'force')

    def test_bundle_and_source_keep_the_extension_and_media_type(self):
        name, data = SAMPLES['docx']
        job = self.service.submit(io.BytesIO(data), name)
        worker = Worker(self.settings)
        row = worker.claim()
        scratch = worker.storage.root / 'attempts' / row['attempt_id']
        (scratch / 'original').mkdir(parents=True)
        write(scratch / 'original', 'document.md', b'converted')
        write_json(scratch, 'engine.json', {'inventory': [identify(scratch / 'original', 'document.md')], 'inventory_complete': True})
        worker.publish(row, scratch, {'test': 'mock'}, {'elapsed_seconds': 0})
        worker.storage.close()
        with zipfile.ZipFile(self.service.bundle(job['id'])) as archive:
            self.assertEqual(sorted(archive.namelist()), ['manifest.json', 'result/document.md', 'source/document.docx'])
            self.assertEqual(archive.read('source/document.docx'), data)
        client = TestClient(create_app(self.settings), base_url=f'http://127.0.0.1:{self.settings.port}')
        response = client.get('/api/v1/runs/%s/source' % job['id'])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'],
                         'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        self.assertEqual(response.headers['content-disposition'], 'attachment; filename="document.docx"')
        self.assertEqual(response.content, data)

    def test_an_upload_of_any_format_is_collected_by_the_recovery(self):            # K4
        root = self.service.storage.root
        token = uuid.uuid4().hex
        for paths, expected in ((['staging/upload-' + token, 'sources/' + token + '.xlsx'], 'sources/' + token + '.xlsx'),
                                (['staging/upload-' + token], 'sources/' + token + '.pdf')):       # written before C
            with self.subTest(paths=paths):
                self.assertEqual(upload_source({'token': token, 'paths': paths}), expected)
        for paths in (['sources/' + token + '.exe'], ['sources/' + token + '.pdf', 'sources/' + token + '.xlsx'],
                      ['sources/' + uuid.uuid4().hex + '.pdf'], ['sources/' + token]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                upload_source({'token': token, 'paths': paths})
        # an xlsx upload interrupted between the rename and the commit of its rows: a dead
        # reservation, the source in sources/ and no document; the recovery removes it
        (root / 'staging' / ('upload-' + token)).mkdir()
        write(root, 'sources/' + token + '.xlsx', SAMPLES['xlsx'][1])
        record = dict(token=token, kind='upload', budget=1 << 20, paths=['staging/upload-' + token, 'sources/' + token + '.xlsx'])
        write(root, 'operations/lease-' + uuid.uuid4().hex + '.json', (json.dumps(record) + '\n').encode())
        with self.service.storage.lock('admission.lock'):
            recover(self.service.storage)
        self.assertFalse((root / 'sources' / (token + '.xlsx')).exists())
        self.assertFalse((root / 'staging' / ('upload-' + token)).exists())
        self.assertEqual(list((root / 'operations').iterdir()), [])

    def test_api_options_session_and_readiness(self):
        client = TestClient(create_app(self.settings), base_url=f'http://127.0.0.1:{self.settings.port}')
        session = client.get('/api/v1/session').json()
        headers = {'origin': f'http://127.0.0.1:{self.settings.port}', 'x-mdf-csrf': session['csrf']}
        formats = {f['format']: f for f in session['formats']}
        self.assertEqual(set(formats), set(FORMATS))
        self.assertEqual((formats['docx']['default'], formats['docx']['alternatives']), ('pandoc', ['marker']))
        self.assertEqual((formats['pdf']['default'], formats['pdf']['alternatives']), ('marker', ['datalab']))
        self.assertEqual(session['datalab']['configured'], False)        # no worker, cap 0
        self.assertIn('leaves this computer', session['datalab']['notice'])
        self.assertIn('training', session['datalab']['notice'])
        self.assertEqual(client.get('/api/v1/readiness').json(),
                         {'ready': False, 'worker': 'Worker absent; archive available', 'converters': None})
        name, data = SAMPLES['docx']
        response = client.post('/api/v1/runs?route=marker&lang=en', content=data, headers={**headers, 'x-filename': name})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()['options']['route'], 'marker')
        for query, code in (('route=nonsense', 'route_invalid'), ('ocr=force', 'route_option_unsupported'),
                            ('force_ocr=true', 'invalid_setting'), ('transit_confirmed=yes', 'invalid_setting')):
            with self.subTest(query=query):
                response = client.post('/api/v1/runs?' + query, content=data, headers={**headers, 'x-filename': name})
                self.assertEqual((response.status_code, response.json().get('code')), (409, code))
                self.assertTrue(response.json()['error'])
        response = client.post('/api/v1/runs', content=b'%PDF-1.4\n', headers={**headers, 'x-filename': 'fake.docx'})
        self.assertEqual((response.status_code, response.json()['code']), (409, 'format_mismatch'))
        # readiness of the worker per converter, and Datalab configured with a key and a cap above 0
        converters = {'marker': {'available': True, 'code': None}, 'datalab': {'available': True, 'code': None}}
        with self.service.storage.connect() as conn:
            conn.execute("INSERT INTO worker VALUES (1,'t',?,?)", (__import__('time').time(),
                         json.dumps({'ready': True, 'message': 'Local worker ready', 'converters': converters})))
        self.assertEqual(client.get('/api/v1/readiness').json(),
                         {'ready': True, 'worker': 'Local worker ready', 'converters': converters})
        self.assertFalse(client.get('/api/v1/session').json()['datalab']['configured'])
        from dataclasses import replace
        capped = TestClient(create_app(replace(self.settings, datalab_cap_usd='0.05')),
                            base_url=f'http://127.0.0.1:{self.settings.port}')
        self.assertEqual(capped.get('/api/v1/session').json()['datalab']['configured'], True)


class WebUiTests(unittest.TestCase):
    """The provisional Web UI in English (MFL032 C4); its behaviour is proved in the browser."""

    def test_english_page_with_the_transit_confirmation_problems_and_limits(self):
        web = Path(__file__).parents[2] / 'markdown_for_llms' / 'web'
        page, script = (web / 'index.html').read_text(), (web / 'app.js').read_text()
        self.assertIn('<html lang="en">', page)
        for italian in ('Converti', 'Scegli', 'Annulla', 'Ripeti', 'Scarica', 'Qualità', 'archivio'):
            self.assertNotIn(italian, page + script)
        self.assertIn('id="transit"', page)
        self.assertIn("datalab && !$('transit').checked", script)                 # no sending without the confirmation
        self.assertIn("transit_confirmed', 'true'", script)
        self.assertIn('manifest.engine.limits', script)                             # PC2
        self.assertIn('issue.page', script)

    def test_routes_of_a_selection_include_the_defaults_of_the_formats(self):          # FD1 of the fix of C
        # the behaviour is proved on app.js by checks/fix-C/ui_routes.js and in the browser
        script = (Path(__file__).parents[2] / 'markdown_for_llms' / 'web' / 'app.js').read_text()
        self.assertIn('let common = [formats[0].default, ...formats[0].alternatives];', script)
        self.assertIn('r === f.default || f.alternatives.includes(r)', script)
        self.assertIn('!formats.every(f => f.default === r)', script)


if __name__ == '__main__':
    unittest.main()
