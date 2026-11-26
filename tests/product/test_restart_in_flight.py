"""The worker stopped or killed while a child of a conversion runs, on real containers (MFL051, plan r2.1 App. F.2).

A worker process claims a job and starts one long child of the engine through the real callers (`tools.run` for a
`tool` child, `isolation.run` for a `light` and a `marker` child); the test then ends that process:

- SIGTERM, as compose stops the service: the worker asks the engine to stop, the child ends, its end is attested,
  the job is `interrupted` by the worker itself and its commitment released;
- SIGKILL: the container is still running (its supervisor ends it only when its lease is not renewed for thirty
  seconds). The next worker finds the attempt, by the prefix of the names of its containers, BEFORE its first
  claim: it kills and removes what is left, the job is `interrupted`, the indicator of the attempt is 1, the
  commitment is released and the archive is coherent. While the launcher does not answer nothing is released.

The names of the containers of an attempt have one schema for every kind, `mdf-eng-<attempt[:16]>-<kind>-<n>`,
so the reconciliation finds a tool as it finds the Marker child: before MFL051 the container of a tool had not the
prefix the reconciliation looked for.

Needs the launcher of the suite (`make test`): without one it fails, saying so.
"""
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from markdown_for_llms.engine import containers
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.storage import StorageError
from markdown_for_llms.worker import Unattested, Worker

PDF = b'%PDF-1.4\nsynthetic restart contract\n'
# The worker process: the engine is replaced by one long child of the kind asked, started as the engine starts it.
DRIVER = r'''
import json, signal, sys
from pathlib import Path
from markdown_for_llms import worker as worker_module
from markdown_for_llms.engine import containers, isolation, tools
from markdown_for_llms.engine.codes import EngineError
from markdown_for_llms.settings import Settings

data, kind = sys.argv[1], sys.argv[2]


def convert(source, out, request=None, limits=None, work=None, observer=None, **more):
    attempt, work = request["attempt"], Path(work)
    work.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "tool":
            (work / "endless.ps").write_bytes(b"%!PS\n{ 1 pop } loop\n")
            tools.run("gs", ["-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=nullpage", tools.In(work / "endless.ps")],
                      work / "tools", timeout=300, watch=tools.Watch(64, observer=observer, attempt=attempt, poll_seconds=0.2))
        else:
            (work / "child").mkdir()
            (work / "child" / "steps.json").write_text(json.dumps([{"do": "sleep", "seconds": 300}]))
            isolation.run(work / "child", containers.PROBE_MODULE, ["container"], timeout=300, observer=observer,
                          attempt=attempt, kind=kind, poll_seconds=0.2)
        code = "engine_failed"
    except EngineError as exc:
        code = exc.code.value
    return {"status": "failed", "error": {"code": code, "fatal": False}}


worker_module.convert = convert
worker_module.availability = lambda **kw: {}
w = worker_module.Worker(Settings(data_root=data, model_root=containers.MODEL_ROOT))
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: setattr(w, "stop", True))          # as `mdf worker` does
row = w.claim()
print("claimed " + row["attempt_id"], flush=True)
w.process(row)
print("ended", flush=True)
'''


class RestartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_root=self.tmp.name + '/data', model_root=containers.MODEL_ROOT)
        self.service = Service(self.settings)
        self.addCleanup(self.service.storage.close)
        self.launcher = containers.launcher()
        try:
            self.launcher.hello()
        except (containers.Unreachable, containers.Refused) as exc:
            self.fail("no launcher (%s): this test needs real containers, run it with `make test`" % exc)

    def start(self, kind):
        """A worker process with a child of `kind` running; returns (process, job, attempt, name of the container)."""
        job = self.service.submit(io.BytesIO(PDF + kind.encode()), kind + '.pdf')
        proc = subprocess.Popen([sys.executable, '-I', '-B', '-c', DRIVER, self.settings.data_root, kind],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        self.addCleanup(lambda: (proc.stdout.close(), proc.stderr.close()))
        line = proc.stdout.readline()
        self.assertTrue(line.startswith('claimed '), proc.stderr.read() if proc.poll() is not None else line)
        attempt = line.split()[1]
        prefix = containers.prefix_of(attempt)
        self.addCleanup(self.clean, prefix)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            running = [c['name'] for c in self.launcher.list(prefix) if c['state'] == 'running']
            if running:
                self.assertRegex(running[0], r'^%s-%s-\d+$' % (prefix, kind))       # one schema, the kind in the name
                return proc, job, attempt, running[0]
            self.assertIsNone(proc.poll(), 'the worker ended before its child ran')
            time.sleep(0.2)
        self.fail('no child of the attempt ran')

    def clean(self, prefix):
        containers.stop_attempt_children(self.launcher, prefix)
        containers.sweep(self.launcher, containers.WORK_ROOT, prefix=prefix, min_age=0)

    def left(self, attempt):
        prefix = containers.prefix_of(attempt)
        calls = Path(containers.WORK_ROOT) / 'call'
        folders = sorted(n for n in os.listdir(calls) if n.startswith(prefix + '-')) if calls.is_dir() else []
        return [c['name'] for c in self.launcher.list(prefix)], folders

    def records(self):
        return sorted(json.loads(p.read_text())['token'] for p in (self.service.storage.root / 'operations').glob('*.json'))

    def recorded(self, job):
        row = self.service.storage.get(job['id'])
        return row['state'], row['scope_recorded'], json.loads(row['child']) if row['child'] else None

    def test_sigterm_the_worker_stops_its_child_and_attests_its_end(self):
        for kind in ('tool', 'light', 'marker'):
            with self.subTest(kind=kind):
                proc, job, attempt, name = self.start(kind)
                self.assertEqual(self.recorded(job), ('running', 1, {'container': name}))    # named before it was created
                proc.send_signal(signal.SIGTERM)
                self.assertEqual(proc.wait(timeout=60), 0, proc.stderr.read())
                state, indicator, _ = self.recorded(job)
                self.assertEqual((state, indicator), ('interrupted', 1))                    # a declared outcome
                self.assertEqual(self.left(attempt), ([], []))                              # attested before the process ended
                self.assertEqual(self.records(), [])

    def test_sigkill_the_next_worker_removes_what_is_left_before_its_first_claim(self):
        for kind in ('tool', 'light', 'marker'):
            with self.subTest(kind=kind):
                proc, job, attempt, name = self.start(kind)
                waiting = self.service.submit(io.BytesIO(PDF + b'waiting' + kind.encode()), 'waiting.pdf')
                proc.kill()
                proc.wait(timeout=30)
                # nothing stopped the container, and the job still looks running: the commitment of the attempt is on disk
                self.assertEqual([c['state'] for c in self.launcher.list(containers.prefix_of(attempt))], ['running'])
                self.assertEqual((self.recorded(job), self.records()), (('running', 1, {'container': name}), [attempt]))
                # while the launcher does not answer, nothing is released and nothing is claimed
                def refused(path, timeout):
                    raise ConnectionRefusedError(111, 'no launcher')
                with mock.patch.object(containers, 'launcher', lambda: containers.Launcher(connect=refused)):
                    restarted = Worker(self.settings)
                    self.addCleanup(restarted.storage.close)
                    with self.assertRaises(Unattested):
                        restarted.reconcile()
                    with self.assertRaisesRegex(StorageError, 'not attested absent'):
                        restarted.run(once=True)
                self.assertEqual((self.recorded(job)[0], self.records(), self.service.get(waiting['id'])['state']),
                                 ('running', [attempt], 'queued'))
                self.assertEqual([c['state'] for c in self.launcher.list(containers.prefix_of(attempt))], ['running'])
                # with the launcher: the container is killed and removed, then the commitment goes, then the job ends
                restarted.reconcile()
                self.assertEqual(self.left(attempt), ([], []))
                self.assertEqual((self.recorded(job)[:2], self.records()), (('interrupted', 1), []))
                # the archive is coherent: the source is there, the partial of the attempt too, and the queue goes on
                row = self.service.storage.get(job['id'])
                self.assertTrue((self.service.storage.root / row['source_path']).is_file())
                self.assertEqual(self.service.get(waiting['id'])['state'], 'queued')
                with self.service.storage.connect() as conn:
                    conn.execute("UPDATE runs SET state='cancelled' WHERE id=?", (waiting['id'],))


if __name__ == '__main__':
    unittest.main()
