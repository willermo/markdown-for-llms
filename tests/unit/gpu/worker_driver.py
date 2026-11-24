"""A worker of the product with a fake provider and a fake engine of the route marker-server, stopped at a barrier so that
a test kills it there with SIGKILL (MFL023 B4, plan r3 §6 point 6).

    python -I -B worker_driver.py CONFIG.json

CONFIG: {"settings": the fields of Settings, "env_file", "state" (the file of `provider_driver.py`), "barrier": the name
of the narrow window, "file": the barrier file (written with the pid of the worker when it is reached), "close": true to
ask for the disarm after the job, "follower": optional, {"lines": the lines of the worker log, "file": a file that gets
the path of the log}: a fake follower of the worker logs that writes them at its start (MFL023 B10)}. The worker holds
`worker.lock` as `Worker.run` does; it runs the recovery, the reconciliation and its loop, with the real clocks. The
checkout comes first on sys.path, as in the other drivers.
"""
import json
import os
from pathlib import Path
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

from markdown_for_llms.bench import serverspend  # noqa: E402
from markdown_for_llms.gpu import lifecycle  # noqa: E402
from markdown_for_llms import gpu, worker as worker_mod  # noqa: E402
from markdown_for_llms.settings import Settings  # noqa: E402
from markdown_for_llms.worker import Worker  # noqa: E402
import gpu_harness  # noqa: E402

cfg = json.loads(Path(sys.argv[1]).read_text())
name, barrier = cfg["barrier"], cfg["file"]


def block():
    Path(barrier).write_text(str(os.getpid()))
    while True:
        time.sleep(3600)


def after(target, attr, when=lambda *a, **k: True):
    real = getattr(target, attr)

    def wrapped(*a, **k):
        out = real(*a, **k)
        if when(*a, **k):
            block()
        return out
    setattr(target, attr, wrapped)


def before(target, attr):
    real = getattr(target, attr)

    def wrapped(*a, **k):
        block()
        return real(*a, **k)
    setattr(target, attr, wrapped)


def during(observer):
    if name == "after_reserve":
        block()


if name == "before_arm_row":
    from markdown_for_llms.gpu import session as cli_session
    before(cli_session, "take_lock")
elif name == "before_arm":
    after(lifecycle.Session, "start_clock")
elif name == "armed_before_reserve":
    after(lifecycle.Manager, "prepare")
elif name == "after_complete":
    after(lifecycle.Manager, "done")
elif name == "publishing":
    before(Worker, "complete_publication")
elif name == "linger":
    real_poll = lifecycle.Manager.poll

    def poll(self):
        if self.session is not None and self.session.state.snapshot()[0] == "linger":
            block()
        return real_poll(self)
    lifecycle.Manager.poll = poll
elif name == "after_zero_before_row":
    before(serverspend, "disarm")
elif name == "after_row_before_lock":
    after(serverspend, "disarm")

class FakeFollower:
    """The follower of the worker logs as the session uses it (`start`, `stop`, `path`), writing its lines at once."""

    def __init__(self, path, spec):
        self.path, self.spec = Path(path), spec

    def start(self):
        self.path.write_text("".join(line + "\n" for line in self.spec["lines"]))
        Path(self.spec["file"]).write_text(str(self.path))
        return self

    def stop(self):
        return {"lines": len(self.spec["lines"])}


follower = None
if cfg.get("follower"):
    def follower(session, path):
        return FakeFollower(path, cfg["follower"])

settings = Settings(**cfg["settings"])
clocks = lifecycle.Clocks()
w = Worker(settings, env_file=cfg["env_file"],
           gpu={"argv": [sys.executable, "-I", "-B", str(HERE / "provider_driver.py"), cfg["state"]], "clocks": clocks,
                "follower": follower})
worker_mod.convert = gpu_harness.gpu_convert(clocks, run_s=5, during=during)
with w.exclusive():
    w.gpu.recover()
    w.reconcile()
    asked = False
    while True:
        w.gpu.poll()
        row = w.claim()
        if row:
            w.process(row)
        elif cfg.get("close") and not asked and w.gpu.session is not None:
            now = time.time()
            with w.storage.connect() as conn:
                conn.execute("INSERT INTO gpu_requests VALUES (?,?,?,?,'pending',NULL,?,?)",
                             (uuid.uuid4().hex, "local", w.gpu.session.name, "disarm", now, now))
            asked = True
        time.sleep(0.05)
