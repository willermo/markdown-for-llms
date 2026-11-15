"""The worker on the conversion engine (MFL032 tranche B, Δ6): reconciliation of the child record
(K7, PB8), the manifest summary of the engine result (K9, PB9), readiness, and one real PDF from
the archive through the worker, Marker 2.2.0 and the download (12.13, PB10)."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from markdown_for_llms.files import json_bytes, read_json
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import StorageError
from markdown_for_llms.worker import Worker, engine_issues, summarize

from engine_fake import fake_convert

PDF = b'%PDF-1.4\nsynthetic worker contract\n'


def problems(n, page=None):
    return [{"code": "media_furniture", "severity": "warning", "page": page, "detail": "cornice " + "x" * 200}
            for _ in range(n)]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + '/data', model_root=self.tmp.name + '/models')
        self.service = Service(self.settings)
        self.addCleanup(self.service.storage.close)

    def run_job(self, engine, overrides=None):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf', overrides)
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        with patch('markdown_for_llms.worker.convert', side_effect=engine):
            worker.process(worker.claim())
        return self.service.get(job['id']), self.service.storage.get(job['id'])


class ManifestTests(Base):
    def test_5000_problems_fit_the_manifest_and_keep_their_severity(self):               # K9, PB9
        listed = [{"code": "media_lost", "severity": "incomplete", "page": 7, "path": "Pictures/a.png"}] + problems(1999, 3)
        engine = fake_convert(problems=listed, counts={"incomplete": 1, "warning": 4999})
        job, row = self.run_job(engine)
        self.assertEqual((job['state'], job['bundle']), ('completed', 'incomplete'))
        result = self.service.storage.root / row['result_path']
        body = (result / 'manifest.json').read_bytes()
        self.assertLess(len(body), 1048576)
        manifest = json.loads(body)
        issues = manifest['checks']['issues']
        self.assertEqual(len(issues), 201)                       # 200 listed and one line for the rest
        self.assertEqual(issues[0], {"code": "incomplete", "reason": "media lost on page 7: Pictures/a.png",
                                     "path": "Pictures/a.png", "page": 7})
        self.assertIn("4800 more problems", issues[-1]['reason'])
        self.assertEqual(manifest['engine']['problem_counts'], {"incomplete": 1, "warning": 4999})
        self.assertEqual(manifest['engine']['inventory'], [{k: a[k] for k in ('path', 'bytes', 'sha256')}
                                                           for a in manifest['artifacts']])
        self.assertEqual(len(json.loads((result / 'engine-result.json').read_text())['problems']), 2000)

    def test_severity_comes_from_the_counts_before_the_truncation(self):                 # PB9
        # the incomplete problem is beyond the listed ones: the bundle is incomplete all the same
        engine = fake_convert(problems=problems(300), counts={"incomplete": 1, "warning": 300})
        job, _ = self.run_job(engine)
        self.assertEqual(job['bundle'], 'incomplete')

    def test_warnings_only_make_the_bundle_unchecked(self):
        job, _ = self.run_job(fake_convert(problems=problems(2, 1)))
        self.assertEqual(job['bundle'], 'unchecked')
        job, _ = self.run_job(fake_convert())
        self.assertEqual(job['bundle'], 'ready')

    def test_summary_is_bounded(self):
        result = {"status": "complete", "problems": problems(2000), "problem_counts": {"warning": 2000},
                  "pages": {"count": 300, "classes": {"native": 300}, "pages": [{"page": i} for i in range(300)]},
                  "corrections": {"counts": {"link_unlinked": 3}, "items": [{"x": 1}] * 2000},
                  "route": {"converter": "marker", "steps": [{"tool": "t"}] * 50}}
        summary = summarize(result, [])
        self.assertEqual((len(summary['problems']), summary['pages'], summary['corrections']),
                         (200, {"count": 300, "classes": {"native": 300}}, {"link_unlinked": 3}))
        self.assertNotIn('steps', summary['route'])
        self.assertLess(len(json_bytes(summary)), 200_000)
        issues, counts = engine_issues(summary)
        self.assertEqual((len(issues), counts), (201, {"warning": 2000}))


class ReconciliationTests(Base):
    def interrupted_row(self, child):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        row = worker.claim()
        worker.tick(row, child)
        with self.service.storage.connect() as conn:          # the worker died: its token is gone
            conn.execute("UPDATE runs SET token='dead-worker' WHERE id=?", (row['id'],))
        return job

    def test_scope_of_a_dead_attempt_is_stopped(self):                                   # K7
        job = self.interrupted_row({"pid": 999999999, "scope": "mdf-eng-abc-marker-1.scope"})
        with patch('markdown_for_llms.worker.isolation.stop_scope', return_value='absent') as stop:
            Worker(self.settings).reconcile()
        stop.assert_called_once_with('mdf-eng-abc-marker-1.scope')
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')

    def test_unreachable_manager_is_not_an_absent_unit(self):                            # PB8
        from markdown_for_llms.engine.isolation import ScopeUnreachable
        job = self.interrupted_row({"pid": 999999999, "scope": "mdf-eng-abc-marker-1.scope"})
        for effect in (ScopeUnreachable("systemctl exit 1"), None):
            with self.subTest(effect=effect):
                kwargs = {'side_effect': effect} if effect else {'return_value': 'active'}
                with patch('markdown_for_llms.worker.isolation.stop_scope', **kwargs), \
                        self.assertRaisesRegex(StorageError, 'not attested|still active'):
                    Worker(self.settings).reconcile()
                self.assertEqual(self.service.get(job['id'])['state'], 'running')

    def test_scopes_of_the_attempt_are_stopped_by_prefix(self):                          # HC1
        from markdown_for_llms.engine.isolation import ScopeUnreachable
        job = self.interrupted_row(None)                     # the worker died before any record
        with self.service.storage.connect() as conn:         # an attempt of a previous version: no indicator
            conn.execute('UPDATE runs SET scope_recorded=NULL WHERE id=?', (job['id'],))
        row = self.service.storage.get(job['id'])
        prefix = 'mdf-eng-%s-' % row['attempt_id'][:16]
        with patch('markdown_for_llms.worker.isolation.list_scopes', side_effect=ScopeUnreachable('bus')), \
                self.assertRaisesRegex(StorageError, 'not attested'):
            Worker(self.settings).reconcile()
        self.assertEqual(self.service.get(job['id'])['state'], 'running')
        with patch('markdown_for_llms.worker.isolation.list_scopes', return_value=[prefix + 'marker-2.scope']) as listed, \
                patch('markdown_for_llms.worker.isolation.stop_scope', return_value='absent') as stop:
            Worker(self.settings).reconcile()
        listed.assert_called_once_with(prefix)
        stop.assert_called_once_with(prefix + 'marker-2.scope')
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')

    def test_only_attempts_that_recorded_a_scope_ask_the_user_manager(self):             # indicator of the scope
        from markdown_for_llms.engine.isolation import ScopeUnreachable
        # no scope was ever named (a native or Datalab route, or a death before the first one):
        # the reconciliation does not need the user manager, even when it is unreachable
        job = self.interrupted_row({"container": "mdf-eng-abc-1"})
        self.assertEqual(self.service.storage.get(job['id'])['scope_recorded'], 0)
        with patch('markdown_for_llms.worker.isolation.list_scopes', side_effect=ScopeUnreachable('no bus')) as listed, \
                patch('markdown_for_llms.engine.toolchain.remove_container', return_value=True):
            Worker(self.settings).reconcile()
        listed.assert_not_called()
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')
        # a scope was named, then runs.child was replaced (the raster child of the publication):
        # the indicator stays 1 and the scope is still found by its prefix
        job = self.interrupted_row({"pid": None, "scope": "mdf-eng-x-marker-1.scope"})
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        row = self.service.storage.get(job['id'])
        with self.service.storage.connect() as conn:
            conn.execute("UPDATE runs SET token=? WHERE id=?", (worker.token, job['id']))
        worker.tick(dict(row, token=worker.token), {"pid": 999999999})
        with self.service.storage.connect() as conn:
            conn.execute("UPDATE runs SET token='dead-worker' WHERE id=?", (job['id'],))
        row = self.service.storage.get(job['id'])
        self.assertEqual((row['scope_recorded'], json.loads(row['child'])), (1, {"pid": 999999999}))
        prefix = 'mdf-eng-%s-' % row['attempt_id'][:16]
        with patch('markdown_for_llms.worker.isolation.list_scopes', side_effect=ScopeUnreachable('no bus')):
            with self.assertRaisesRegex(StorageError, 'not attested'):
                Worker(self.settings).reconcile()
        with patch('markdown_for_llms.worker.isolation.list_scopes', return_value=[prefix + 'marker-1.scope']) as listed, \
                patch('markdown_for_llms.worker.isolation.stop_scope', return_value='absent') as stop:
            Worker(self.settings).reconcile()
        stop.assert_called_once_with(prefix + 'marker-1.scope')
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')

    def test_a_queued_run_of_a_previous_version_gets_the_indicator_when_claimed(self):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        with self.service.storage.connect() as conn:
            conn.execute('UPDATE runs SET scope_recorded=NULL WHERE id=?', (job['id'],))
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        self.assertEqual(worker.claim()['scope_recorded'], 0)

    def test_the_scope_is_recorded_before_the_child_has_a_pid(self):                    # HC1
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        seen = []

        def record(observer):
            with self.service.storage.connect() as conn:
                seen.append(json.loads(conn.execute('SELECT child FROM runs WHERE id=?', (job['id'],)).fetchone()[0]))
        engine = fake_convert(events=[{"event": "child", "module": "m", "pid": None, "scope": "mdf-eng-x-marker-1.scope"}],
                              before=record)
        with patch('markdown_for_llms.worker.convert', side_effect=engine):
            worker.process(worker.claim())
        self.assertEqual(seen, [{"scope": "mdf-eng-x-marker-1.scope"}])

    def test_container_of_a_dead_attempt_is_removed(self):
        job = self.interrupted_row({"container": "mdf-eng-abc-1"})
        with patch('markdown_for_llms.engine.toolchain.remove_container', return_value=False):
            with self.assertRaisesRegex(StorageError, 'container not removed'):
                Worker(self.settings).reconcile()
        with patch('markdown_for_llms.engine.toolchain.remove_container', return_value=True):
            Worker(self.settings).reconcile()
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')

    def test_malformed_child_record_is_refused(self):
        self.interrupted_row({"scope": "../../etc.scope"})
        with self.assertRaisesRegex(StorageError, 'Malformed child record'):
            Worker(self.settings).reconcile()


class ReadinessTests(Base):
    AVAILABLE = {"text": {"available": True, "code": None}, "pandoc": {"available": True, "code": None},
                 "djvutxt": {"available": True, "code": None}, "datalab": {"available": False, "code": "datalab_key_missing"}}

    def test_an_unavailable_route_fails_its_job_and_the_queue_goes_on(self):            # C3
        docx = __import__('test_formats_product').SAMPLES['docx'][1]
        pdf_job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        docx_job = self.service.submit(io.BytesIO(docx), 'letter.docx')
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        states = [dict(self.AVAILABLE, marker={"available": False, "code": "weights_missing"}),
                  dict(self.AVAILABLE, marker={"available": True, "code": None})]
        calls = []
        with patch('markdown_for_llms.worker.availability', side_effect=states) as check, \
                patch('markdown_for_llms.worker.convert', side_effect=fake_convert(calls=calls)):
            worker.ready()
            worker.heartbeat()
            worker.process(worker.claim())
            worker.process(worker.claim())
            worker.ready()                                    # held for READY_SECONDS
            self.assertEqual(check.call_count, 1)
            worker._ready_at -= 61
            self.assertTrue(worker.ready()['marker']['available'])
        self.assertEqual(check.call_args.kwargs, {'model_root': self.settings.model_root})
        failed, done = self.service.get(pdf_job['id']), self.service.get(docx_job['id'])
        self.assertEqual(failed['state'], 'failed')
        self.assertTrue(failed['error'].startswith('Conversion failed: route_unavailable (marker: weights_missing)'),
                        failed['error'])
        self.assertEqual(done['state'], 'completed')
        self.assertEqual([(c['request']['route'], c['name']) for c in calls], [(None, 'letter.docx')])
        with self.service.storage.connect() as conn:
            readiness = json.loads(conn.execute('SELECT readiness FROM worker').fetchone()[0])
        self.assertEqual(readiness['converters']['marker'], {"available": False, "code": "weights_missing"})
        self.assertIn('routes unavailable: datalab (datalab_key_missing), marker (weights_missing)', readiness['message'])


class DatalabWorkerTests(Base):
    SECRET = 'synthetic-datalab-key-0123456789'

    def test_cli_retry_of_a_datalab_run_confirms_the_transit_again(self):            # FD2 of the fix of C
        """A Datalab run completed through the worker, the engine and the fake transport of
        test_engine_datalab (no network); `mdf retry ID` without the confirmation queues nothing
        and exits 1 with transit_not_confirmed, `--confirm-transit` queues a new attempt."""
        import contextlib
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unit' / 'engine'))
        from engine_test_support import NATIVE_TEXT, make_pdf
        from test_engine_datalab import FakeTransport
        from markdown_for_llms import cli
        from markdown_for_llms.engine import datalab
        from dataclasses import replace
        settings = replace(self.settings, datalab_cap_usd='0.05')
        pdf = make_pdf([{"content": NATIVE_TEXT}, {"content": NATIVE_TEXT}])
        job = self.service.submit(io.BytesIO(pdf), 'remote.pdf', {'route': 'datalab', 'transit_confirmed': True})
        with patch.dict(os.environ, {'DATALAB_API_KEY': self.SECRET}):
            worker = Worker(settings)
        self.addCleanup(worker.storage.close)
        transport = FakeTransport()
        with patch.object(datalab, 'Transport', return_value=transport), patch.object(datalab, 'POLL_SECONDS', 0.01):
            worker.process(worker.claim())
        done = self.service.get(job['id'])
        self.assertEqual((done['state'], transport.methods()[0], transport.methods()[-1]), ('completed', 'POST', 'DELETE'),
                         done.get('error'))
        base = ['--data-root', self.settings.data_root, '--model-root', self.settings.model_root]

        def run(argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main(base + argv)
            return code, json.loads(out.getvalue())
        code, body = run(['retry', job['id']])
        self.assertEqual((code, body.get('code')), (1, 'transit_not_confirmed'), body)
        self.assertEqual(len(self.service.list()), 1)                    # no new attempt
        code, body = run(['retry', job['id'], '--confirm-transit'])
        self.assertEqual(code, 0, body)
        self.assertEqual((body['retry_of'], body['state'], body['options']['transit_confirmed']), (job['id'], 'queued', True))
        self.assertEqual(len(self.service.list()), 2)
        # the same refusal as the API for a run that is not a Datalab one
        local = self.service.submit(io.BytesIO(pdf), 'local.pdf')
        self.service.cancel(local['id'])
        code, body = run(['retry', local['id'], '--confirm-transit'])
        self.assertEqual((code, body.get('code')), (1, 'route_option_unsupported'), body)

    def test_the_key_leaves_the_environment_and_the_transit_is_the_one_of_the_job(self):
        from dataclasses import replace
        settings = replace(self.settings, datalab_cap_usd='0.05')
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf', {'route': 'datalab', 'transit_confirmed': True,
                                                             'datalab_mode': 'balanced'})
        with patch.dict(os.environ, {'DATALAB_API_KEY': self.SECRET}):
            worker = Worker(settings)
            self.assertNotIn('DATALAB_API_KEY', os.environ)          # no child of the worker inherits it
        self.addCleanup(worker.storage.close)
        self.assertEqual(worker.datalab_state(), {"available": True, "code": None})
        self.assertEqual(Worker(self.settings).datalab_state()['code'], 'datalab_key_missing')
        calls = []
        with patch('markdown_for_llms.worker.convert', side_effect=fake_convert(calls=calls)):
            worker.process(worker.claim())
        self.assertEqual(calls[0]['request']['datalab'], {'mode': 'balanced', 'transit_confirmed': True, 'cap_usd': '0.05'})
        self.assertIs(calls[0]['datalab_key'], worker.datalab_key)
        self.assertEqual(repr(worker.datalab_key), 'Key(***)')
        row = self.service.storage.get(job['id'])
        self.assertEqual(row['state'], 'completed')
        self.assertTrue(json.loads(row['options'])['transit_confirmed'])
        for path in Path(self.settings.data_root).rglob('*'):
            if path.is_file():
                self.assertNotIn(self.SECRET.encode(), path.read_bytes(), path)
        # a capped worker without a cap above 0 reports Datalab unavailable
        with patch.dict(os.environ, {'DATALAB_API_KEY': self.SECRET}):
            zero = Worker(self.settings)
        self.addCleanup(zero.storage.close)
        self.assertEqual(zero.datalab_state(), {"available": False, "code": "datalab_cap"})


def model_root():
    root = os.environ.get("FULL_CHECK_MODEL_ROOT")
    if not root:
        raise AssertionError("FULL_CHECK_MODEL_ROOT is not set: run the suite from scripts/full-check.sh with the weights "
                             "prepared by `mdf prepare --weights --model-root DIR`")
    return root


class RealWorkerTests(unittest.TestCase):
    """12.13 with PB10: a PDF of the archive through the worker, the engine and Marker 2.2.0
    (weights of FULL_CHECK_MODEL_ROOT), then the bundle downloaded and checked."""

    def test_pdf_through_worker_engine_and_download(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unit' / 'engine'))
        from engine_test_support import FIGURE, NATIVE_TEXT, checker_rgb, make_pdf
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(data_root=tmp + '/data', model_root=model_root(), threads=2, memory_mib=4096,
                                storage_mib=4096, output_mib=256)
            service = Service(settings)
            self.addCleanup(service.storage.close)
            pdf = make_pdf([{"content": NATIVE_TEXT + b" " + FIGURE, "image": (100, 100, checker_rgb(100, 100))}])
            job = service.submit(io.BytesIO(pdf), 'native.pdf')
            worker = Worker(settings)
            self.addCleanup(worker.storage.close)
            worker.ready()
            worker.process(worker.claim())
            final = service.get(job['id'])
            self.assertEqual((final['state'], final['bundle']), ('completed', 'ready'), final.get('error'))
            row = service.storage.get(job['id'])
            result = service.storage.root / row['result_path']
            engine = json.loads((result / 'engine-result.json').read_text())
            self.assertEqual((engine['status'], engine['route']['converter']), ('complete', 'marker'))
            self.assertEqual(read_json(result, 'request.json')['engine_request']['ocr'], 'auto')
            with zipfile.ZipFile(service.bundle(job['id'])) as z:
                names = z.namelist()
                document = next(n for n in names if n.endswith('document.md'))
                self.assertIn("Native text of the synthetic page", z.read(document).decode())
                self.assertTrue(any('/assets/asset-0001.' in n or n.startswith('assets/asset-0001.') for n in names), names)

    def test_docx_through_worker_native_route_and_download(self):                       # C3, 12.1
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unit' / 'engine'))
        from engine_test_support import drawing, make_docx, para, png
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(data_root=tmp + '/data', model_root=model_root(), threads=2, memory_mib=4096,
                                storage_mib=4096, output_mib=256)
            service = Service(settings)
            self.addCleanup(service.storage.close)
            docx = make_docx(Path(tmp) / 'letter.docx', para('Native text of the synthetic letter') + '<w:p><w:r>'
                             + drawing('rId1') + '</w:r></w:p>', media={'image1.png': png()})
            job = service.submit(docx.open('rb'), 'letter.docx')
            worker = Worker(settings)
            self.addCleanup(worker.storage.close)
            self.assertTrue(worker.ready()['pandoc']['available'])
            worker.process(worker.claim())
            final = service.get(job['id'])
            self.assertEqual((final['state'], final['bundle'], final['format']), ('completed', 'ready', 'docx'), final.get('error'))
            engine = json.loads((service.storage.root / service.storage.get(job['id'])['result_path'] / 'engine-result.json').read_text())
            self.assertEqual((engine['route']['converter'], engine['source']['name']), ('pandoc', 'letter.docx'))
            with zipfile.ZipFile(service.bundle(job['id'])) as z:
                names = z.namelist()
                self.assertIn('source/document.docx', names)
                self.assertIn('Native text of the synthetic letter', z.read('result/document.md').decode())
                self.assertIn('result/assets/asset-0001.png', names)


if __name__ == '__main__':
    unittest.main()
