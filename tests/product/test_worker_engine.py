"""The worker on the conversion engine (MFL032 tranche B, Δ6): reconciliation of what a dead attempt
left, its containers, through the launcher (K7, HC1, PB8; MFL051), the manifest summary of the engine
result (K9, PB9), readiness, and one real PDF from the archive through the worker, Marker 2.2.0 and
the download (12.13, PB10). The release rule of B1 is in test_unattested_end.py."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from markdown_for_llms.engine import containers
from markdown_for_llms.files import json_bytes, read_json
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import StorageError
from markdown_for_llms.worker import Worker, engine_issues, summarize

from engine_fake import fake_convert, launcher_fake

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


class TimingsTests(Base):
    def test_the_manifest_keeps_the_stages_and_the_statistics_of_the_pages(self):        # M5 of MFL026, S17
        pages = {"unit": "s", "clock": "monotonic, Marker child", "count": 2000, "median": 1.5, "max": 9.25,
                 "values": [1.5] * 1999 + [9.25]}
        timings = {"total_seconds": 3100.0, "stages": {"models": 4.5, "pages": 3000.0, "render": 2.0}, "pages": pages}
        summary = summarize({"status": "complete", "timings": timings}, [])
        expected = {"total_seconds": 3100.0, "stages": timings["stages"],
                    "pages": {k: v for k, v in pages.items() if k != "values"}}
        self.assertEqual(summary["timings"], expected)
        self.assertEqual(len(timings["pages"]["values"]), 2000)         # the result itself is not changed
        job, row = self.run_job(fake_convert(timings=timings))
        self.assertEqual(job['state'], 'completed')
        result = self.service.storage.root / row['result_path']
        body = (result / 'manifest.json').read_bytes()
        self.assertLess(len(body), 20_000)                              # 2000 page values would not fit here
        self.assertEqual(json.loads(body)['engine']['timings'], expected)
        self.assertEqual(len(json.loads((result / 'engine-result.json').read_text())['timings']['pages']['values']), 2000)
        reopened = Service(self.settings)                               # a new process reads the same job
        self.addCleanup(reopened.storage.close)
        self.assertEqual(reopened.get(job['id'])['state'], 'completed')
        again = reopened.storage.get(job['id'])
        self.assertEqual(json.loads((reopened.storage.root / again['result_path'] / 'manifest.json').read_bytes())
                         ['engine']['timings'], expected)

    def test_timings_of_a_route_without_pages_pass_unchanged(self):
        self.assertEqual(summarize({"timings": {"total_seconds": 2.0}}, [])["timings"], {"total_seconds": 2.0})


class ReconciliationTests(Base):
    """What a dead worker left is a container of the attempt: reconciliation asks the launcher, by the prefix of
    the names of the attempt, and releases nothing until Docker attests that none is left (K7, HC1, PB8)."""

    def setUp(self):
        super().setUp()
        self.launcher = launcher_fake.patched(self, Path(self.tmp.name) / 'volume')

    def interrupted_row(self, child):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        row = worker.claim()
        worker.tick(row, child)
        with self.service.storage.connect() as conn:          # the worker died: its token is gone
            conn.execute("UPDATE runs SET token='dead-worker' WHERE id=?", (row['id'],))
        return job

    def left(self, job, kind='marker', n=1, state='running'):
        """A container the dead attempt of `job` left, as the launcher lists it; returns its name."""
        name = containers.child_name(self.service.storage.get(job['id'])['attempt_id'], kind, n)
        self.launcher.children[name] = {"state": state, "exit": 0, "oom_killed": False}
        return name

    def test_the_container_of_a_dead_attempt_is_removed(self):                          # K7
        job = self.interrupted_row({"container": "placeholder"})
        name = self.left(job)
        Worker(self.settings).reconcile()
        self.assertEqual((self.launcher.ops('kill'), self.launcher.removed), ([('kill', name)], [name]))
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')

    def test_a_launcher_that_does_not_answer_is_not_an_absent_container(self):          # PB8
        job = self.interrupted_row({"container": "placeholder"})
        name = self.left(job, 'tool')
        for effect in ('down', 'unconfirmed'):
            with self.subTest(effect=effect):
                self.launcher.down = effect == 'down'
                self.launcher.unconfirmed = {name} if effect == 'unconfirmed' else set()
                with self.assertRaisesRegex(StorageError, 'not attested absent'):
                    Worker(self.settings).reconcile()
                self.assertEqual(self.service.get(job['id'])['state'], 'running')
        self.launcher.down, self.launcher.unconfirmed = False, set()
        Worker(self.settings).reconcile()
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')

    def test_the_containers_of_the_attempt_are_found_by_prefix(self):                   # HC1
        job = self.interrupted_row(None)                     # the worker died before any record
        with self.service.storage.connect() as conn:         # an attempt of a previous version: no indicator
            conn.execute('UPDATE runs SET scope_recorded=NULL WHERE id=?', (job['id'],))
        names = [self.left(job, 'marker', 2), self.left(job, 'light', 3, state='exited'), self.left(job, 'tool', 7)]
        other = 'mdf-eng-0000000000000000-marker-1'          # of another attempt: never touched
        self.launcher.children[other] = {"state": "running", "exit": 0, "oom_killed": False}
        self.launcher.down = True
        with self.assertRaisesRegex(StorageError, 'not attested absent'):
            Worker(self.settings).reconcile()
        self.assertEqual(self.service.get(job['id'])['state'], 'running')
        self.launcher.down = False
        Worker(self.settings).reconcile()
        prefix = containers.prefix_of(self.service.storage.get(job['id'])['attempt_id'])
        self.assertIn(('list', prefix), self.launcher.calls)
        self.assertEqual(sorted(self.launcher.removed), sorted(names))
        self.assertEqual(sorted(self.launcher.children), [other])
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')

    def test_only_attempts_that_named_a_container_ask_the_launcher(self):                # the indicator of the attempt
        # no container was ever named (the text route, Datalab, or a death before the first child): the
        # reconciliation does not need the launcher, even when it is unreachable
        job = self.interrupted_row(None)
        self.assertEqual(self.service.storage.get(job['id'])['scope_recorded'], 0)
        self.launcher.down = True
        Worker(self.settings).reconcile()
        self.assertEqual(self.launcher.calls, [])
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')
        # a container was named, then runs.child was replaced (another child, or none): the indicator stays 1
        # and what the attempt left is still found by its prefix
        job = self.interrupted_row({"container": "mdf-eng-x-marker-1"})
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        row = self.service.storage.get(job['id'])
        with self.service.storage.connect() as conn:
            conn.execute("UPDATE runs SET token=? WHERE id=?", (worker.token, job['id']))
        worker.tick(dict(row, token=worker.token), None)
        with self.service.storage.connect() as conn:
            conn.execute("UPDATE runs SET token='dead-worker' WHERE id=?", (job['id'],))
        row = self.service.storage.get(job['id'])
        self.assertEqual((row['scope_recorded'], row['child']), (1, None))
        with self.assertRaisesRegex(StorageError, 'not attested absent'):
            Worker(self.settings).reconcile()
        self.launcher.down = False
        name = self.left(job)
        Worker(self.settings).reconcile()
        self.assertEqual(self.launcher.removed, [name])
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')

    def test_a_queued_run_of_a_previous_version_gets_the_indicator_when_claimed(self):
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf')
        with self.service.storage.connect() as conn:
            conn.execute('UPDATE runs SET scope_recorded=NULL WHERE id=?', (job['id'],))
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        self.assertEqual(worker.claim()['scope_recorded'], 0)

    def test_the_container_is_recorded_before_it_is_created_for_every_kind(self):       # HC1
        for event in ({"event": "child", "module": "m", "container": "mdf-eng-x-marker-1"},
                      {"event": "tool", "tool": "pandoc", "container": "mdf-eng-x-tool-2"}):
            with self.subTest(event=event["event"]):
                job = self.service.submit(io.BytesIO(PDF + event["event"].encode()), 'p.pdf')
                worker = Worker(self.settings)
                self.addCleanup(worker.storage.close)
                seen = []

                def record(observer):
                    with self.service.storage.connect() as conn:
                        row = conn.execute('SELECT child, scope_recorded FROM runs WHERE id=?', (job['id'],)).fetchone()
                    seen.append((json.loads(row[0]), row[1]))
                with patch('markdown_for_llms.worker.convert', side_effect=fake_convert(events=[event], before=record)):
                    worker.process(worker.claim())
                self.assertEqual(seen, [({"container": event["container"]}, 1)])

    def test_a_record_of_a_previous_version_is_reconciled_by_the_prefix_of_its_attempt(self):
        """A row written before MFL051 may name a scope or a pid: neither exists in the containers. Its attempt is
        still attested by the prefix of its names."""
        job = self.interrupted_row({"pid": 999999999, "scope": "mdf-eng-abc-marker-1.scope"})
        self.assertEqual(self.service.storage.get(job['id'])['scope_recorded'], 1)
        Worker(self.settings).reconcile()
        prefix = containers.prefix_of(self.service.storage.get(job['id'])['attempt_id'])
        self.assertEqual(self.launcher.ops('list'), [('list', prefix), ('list', prefix)])
        self.assertEqual(self.service.get(job['id'])['state'], 'interrupted')


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
        self.assertEqual(check.call_args.kwargs, {'model_root': self.settings.model_root, 'keys_file': None})
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
        keys = Path(self.tmp.name) / 'keys.env'
        keys.write_text('DATALAB_API_KEY=%s\n' % self.SECRET)
        keys.chmod(0o600)
        worker = Worker(settings, env_file=keys)
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

    def test_the_key_comes_from_the_keys_file_and_the_transit_is_the_one_of_the_job(self):
        """MFL051: the key is read by name from the keys file at every use, never from the environment, out of
        which the worker still takes the name so that no child inherits it."""
        from dataclasses import replace
        settings = replace(self.settings, datalab_cap_usd='0.05')
        job = self.service.submit(io.BytesIO(PDF), 'p.pdf', {'route': 'datalab', 'transit_confirmed': True,
                                                             'datalab_mode': 'balanced'})
        keys = Path(self.tmp.name) / 'keys.env'
        keys.write_text('DATALAB_API_KEY=%s\n' % self.SECRET)
        keys.chmod(0o600)
        with patch.dict(os.environ, {'DATALAB_API_KEY': 'from-the-environment'}):
            worker = Worker(settings, env_file=keys)
            self.assertNotIn('DATALAB_API_KEY', os.environ)          # no child of the worker inherits it
            without = Worker(settings)
            self.addCleanup(without.storage.close)
        self.addCleanup(worker.storage.close)
        self.assertEqual(worker.datalab_state(), {"available": True, "code": None})
        self.assertEqual(without.datalab_state()['code'], 'datalab_key_missing')    # the environment gave nothing
        self.assertEqual(Worker(self.settings).datalab_state()['code'], 'datalab_key_missing')
        calls = []
        with patch('markdown_for_llms.worker.convert', side_effect=fake_convert(calls=calls)):
            worker.process(worker.claim())
        self.assertEqual(calls[0]['request']['datalab'], {'mode': 'balanced', 'transit_confirmed': True, 'cap_usd': '0.05'})
        self.assertEqual(calls[0]['datalab_key'].header(), {'X-API-Key': self.SECRET})
        self.assertEqual(repr(calls[0]['datalab_key']), 'Key(***)')
        row = self.service.storage.get(job['id'])
        self.assertEqual(row['state'], 'completed')
        self.assertTrue(json.loads(row['options'])['transit_confirmed'])
        for path in Path(self.settings.data_root).rglob('*'):
            if path.is_file():
                self.assertNotIn(self.SECRET.encode(), path.read_bytes(), path)
        # a capped worker without a cap above 0 reports Datalab unavailable
        zero = Worker(self.settings, env_file=keys)
        self.addCleanup(zero.storage.close)
        self.assertEqual(zero.datalab_state(), {"available": False, "code": "datalab_cap"})


def model_root():
    root = os.environ.get("FULL_CHECK_MODEL_ROOT")
    if not root:
        raise AssertionError("FULL_CHECK_MODEL_ROOT is not set: run the suite with `make test`, with the weights "
                             "prepared in the models volume (`make prepare`)")
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
