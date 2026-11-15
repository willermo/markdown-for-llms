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
    def test_marker_route_unavailable_is_visible_and_checked_again_later(self):
        worker = Worker(self.settings)
        self.addCleanup(worker.storage.close)
        states = [{"marker": {"available": False, "code": "weights_missing"}},
                  {"marker": {"available": True, "code": None}}]
        with patch('markdown_for_llms.worker.availability', side_effect=states) as check:
            with self.assertRaisesRegex(ValueError, 'weights_missing'):
                worker.ready()
            worker.ready()
            worker.ready()                                    # held for READY_SECONDS
        self.assertEqual(check.call_count, 2)
        self.assertEqual(check.call_args.kwargs, {'model_root': self.settings.model_root, 'only': ('marker',)})


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


if __name__ == '__main__':
    unittest.main()
