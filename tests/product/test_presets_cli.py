"""The conversion presets from the command line (MFL023 A5): `mdf presets list|show|add|remove|default` and
`mdf submit --preset`, through the same services as the API."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from markdown_for_llms.cli import main

PDF = b'%PDF-1.4\nsynthetic contract input; not a valid conversion fixture\n'


class PresetsCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf = self.root / 'a.pdf'
        self.pdf.write_bytes(PDF)

    def mdf(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(['--data-root', str(self.root / 'data'), '--model-root', str(self.root / 'models'), *argv])
        text, values, decoder, at = out.getvalue(), [], json.JSONDecoder(), 0
        while text[at:].strip():                      # one indented JSON value per emit
            at += len(text[at:]) - len(text[at:].lstrip())
            value, at = decoder.raw_decode(text, at)
            values.append(value)
        return code, values

    def stored(self, run_id):
        with sqlite3.connect(self.root / 'data' / 'archive.sqlite3') as conn:
            value = conn.execute('SELECT options FROM runs WHERE id=?', (run_id,)).fetchone()[0]
        conn.close()
        return json.loads(value)

    def test_list_show_add_default_remove(self):
        code, (listed,) = self.mdf('presets', 'list')
        self.assertEqual((code, sorted(p['name'] for p in listed['presets'])),
                         (0, ['batch', 'datalab', 'local-fast', 'quality-now']))
        code, (shown,) = self.mdf('presets', 'show', 'batch')
        self.assertEqual((shown['options']['gpu_preset'], shown['available'], shown['code']),
                         ('runpod-serverless-volume-dc', False, 'gpu_preset_not_enabled'))
        code, (added,) = self.mdf('presets', 'add', 'scans', '--route', 'marker', '--ocr', 'force')
        self.assertEqual((code, added['options']['ocr'], added['builtin']), (0, 'force', False))
        code, (refused,) = self.mdf('presets', 'add', 'local-fast', '--route', 'marker')
        self.assertEqual((code, refused['code']), (1, 'preset_reserved'))
        code, (refused,) = self.mdf('presets', 'add', 'gpu', '--route', 'marker-server', '--server-mode', 'balanced',
                                    '--gpu-preset', 'runpod-serverless-volume-eu-se-1')
        self.assertEqual((code, refused['code']), (1, 'gpu_preset_superseded'))
        self.assertEqual(self.mdf('presets', 'default', 'scans')[1], [{'default': 'scans'}])
        self.assertEqual(self.mdf('presets', 'list')[1][0]['default'], 'scans')
        code, (after,) = self.mdf('presets', 'remove', 'scans')
        self.assertEqual((code, after['default']), (0, None))
        self.assertEqual(self.mdf('presets', 'default', '--clear')[1], [{'default': None}])
        self.assertEqual(self.mdf('presets', 'show', 'scans')[1][0]['code'], 'preset_unknown')

    def test_submit_with_a_preset(self):
        self.mdf('presets', 'add', 'scans', '--route', 'marker', '--ocr', 'force')
        code, (job,) = self.mdf('submit', str(self.pdf), '--preset', 'scans', '--lang', 'it')
        self.assertEqual(code, 0, job)
        self.assertEqual(self.stored(job['id']), {'route': 'marker', 'ocr': 'force', 'lang': 'it', 'datalab_mode': None,
                                                  'transit_confirmed': False, 'preset': 'scans'})
        code, (job,) = self.mdf('submit', str(self.pdf))
        self.assertEqual(sorted(self.stored(job['id'])), ['datalab_mode', 'lang', 'ocr', 'route', 'transit_confirmed'])
        for extra, error in ((('--preset', 'quality-now'), 'gpu_preset_not_enabled'),
                             (('--preset', 'local-fast', '--route', 'marker'), 'invalid_setting'),
                             (('--preset', 'local-fast', '--ocr', 'force'), 'invalid_setting'),
                             (('--preset', 'datalab'), 'transit_not_confirmed'),
                             (('--preset', 'nope'), 'preset_unknown')):
            with self.subTest(extra=extra):
                code, (out,) = self.mdf('submit', str(self.pdf), *extra)
                self.assertEqual((code, out['code']), (1, error))


if __name__ == '__main__':
    unittest.main()
