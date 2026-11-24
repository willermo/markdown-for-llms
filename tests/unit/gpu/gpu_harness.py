"""The harness of the tests of the GPU sessions of the product (MFL023 B4): an instance with its servers.json, its env file
(synthetic keys, mode 0600), its spend ledger outside the archive and a fake provider (`provider_driver.py`) for the
children of `opchild`; clocks given to the worker, the wall clock and the monotonic one apart; a fake engine of the route
marker-server that hands the event `server-wake` to the observer as the engine does and writes what the engine writes."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

from markdown_for_llms.bench import serverspend, spend
from markdown_for_llms.engine import servers
from markdown_for_llms.engine.codes import Code, EngineError
from markdown_for_llms.files import identify, write, write_json
from markdown_for_llms.services import Service
from markdown_for_llms.settings import Settings
from markdown_for_llms.worker import Worker

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "provider_driver.py"
RUNPOD_URL = "https://ep1.api.runpod.ai"
MODAL_URL = "https://ws--mdf-surya-serve.modal.run"
PDF = b'%PDF-1.4\nsynthetic contract input; not a valid conversion fixture\n'
KEY = "rk-b4-synthetic-key/0123+456789=="
JOBS = {"runpod": {"route": "marker-server", "server_mode": "balanced", "gpu_preset": "runpod-serverless-volume-dc",
                   "transit_confirmed": True},
        "modal": {"route": "marker-server", "server_mode": "balanced", "gpu_preset": "modal-base",
                  "transit_confirmed": True}}


class FakeClocks:
    """`sleep` moves both clocks (time that passes) with a tiny real pause; `idle` waits in real time and moves nothing;
    `jump` moves the wall clock alone (a suspend, a step of the clock); `advance` moves both."""

    def __init__(self, wall=1_800_000_000.0, mono=10_000.0):
        self.lock = threading.Lock()
        self.t_wall, self.t_mono = wall, mono

    def wall(self):
        with self.lock:
            return self.t_wall

    def mono(self):
        with self.lock:
            return self.t_mono

    def advance(self, s):
        with self.lock:
            self.t_wall += s
            self.t_mono += s

    def jump(self, s):
        with self.lock:
            self.t_wall += s

    def sleep(self, s):
        self.advance(s)
        time.sleep(0.002)

    def idle(self, s):
        time.sleep(min(s, 0.01))


def wait_for(predicate, within=20.0, what="a condition"):
    end = time.monotonic() + within
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("not reached within %s s: %s" % (within, what))


class Instance:
    """One instance of the product with the two GPU presets enabled."""

    def __init__(self, test, provider="runpod", session_cap="0.40", provider_caps=("3", "2"), servers_doc=None,
                 state=None, env=True, max_pages=50):
        self.test = test
        self.root = Path(tempfile.mkdtemp(prefix="mdf-gpu-"))
        test.addCleanup(shutil.rmtree, self.root, True)
        self.provider = provider
        self.ledger_dir = self.root / "ledger"
        self.servers = self.root / "servers.json"
        self.servers.write_text(json.dumps(servers_doc or {"default": "rp", "servers": {
            "rp": {"kind": "runpod", "preset": "runpod-serverless-volume-dc", "resource": "mdf-surya", "url": RUNPOD_URL},
            "md": {"kind": "modal", "preset": "modal-base", "resource": "mdf-surya", "url": MODAL_URL}}}))
        self.env = self.root / ".env"
        if env:
            self.env.write_text("RUNPOD_API_KEY=%s\nMODAL_PROXY_TOKEN_ID=wk-b4-id\nMODAL_PROXY_TOKEN_SECRET=ws-b4-secret\n"
                                % KEY)
            self.env.chmod(0o600)
        self.state = self.root / "provider.json"
        self.state.write_text(json.dumps(dict({"provider": provider, "leak": KEY}, **(state or {}))))
        self.settings = Settings(data_root=str(self.root / "data"), model_root=str(self.root / "models"),
                                 gpu_presets="modal-base,runpod-serverless-volume-dc", servers=str(self.servers),
                                 gpu_ledger_dir=str(self.ledger_dir), gpu_cap_usd_modal=provider_caps[0],
                                 gpu_cap_usd_runpod=provider_caps[1], gpu_session_cap_usd=session_cap,
                                 max_pages=max_pages)
        self.clocks = FakeClocks()
        self.service = Service(self.settings)
        test.addCleanup(self.service.storage.close)
        self.workers = []

    def worker(self, follower=None):
        w = Worker(self.settings, env_file=str(self.env) if self.env.exists() else None,
                   gpu={"argv": [sys.executable, "-I", "-B", str(DRIVER), str(self.state)], "clocks": self.clocks,
                        "follower": follower})
        self.workers.append(w)
        self.test.addCleanup(self.stop, w)
        return w

    def stop(self, w):
        w.stop = True
        try:
            w.gpu.shutdown()
        except Exception:
            pass
        w.storage.close()

    def submit(self, options=None, name="a.pdf"):
        return self.service.submit(io.BytesIO(PDF), name, dict(options or JOBS[self.provider]))

    def provider_state(self):
        return json.loads(self.state.read_text())

    def change(self, **values):
        from provider_driver import Locked
        with Locked(self.state) as s:
            s.update(values)

    def calls(self):
        return [c[0] for c in self.provider_state().get("calls", [])]

    def rows(self, kind=None):
        rows = spend.read(self.ledger_dir / spend.LEDGER)
        return [r for r in rows if kind is None or r.get("kind") == kind]

    def summary(self, provider=None):
        return serverspend.summary(self.rows(), provider or self.provider, self.ledger_dir / spend.LEDGER)

    def sessions(self):
        with self.service.storage.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM gpu_sessions ORDER BY created")]

    def run(self, run_id):
        return self.service.storage.get(run_id)

    def request(self, kind, preset=None):
        import uuid
        now = time.time()
        with self.service.storage.connect() as conn:
            conn.execute("INSERT INTO gpu_requests (id, owner, gpu_preset, kind, state, created, updated) VALUES "
                         "(?,?,?,?,'pending',?,?)", (uuid.uuid4().hex, "local", preset or JOBS[self.provider]["gpu_preset"],
                                                     kind, now, now))

    def settle(self, w, within=30.0):
        """Wait for the session of `w` to reach a final state and the worker to forget it."""
        s = w.gpu.session
        if s is not None and s.thread is not None:
            s.thread.join(within)
            if s.thread.is_alive():
                raise AssertionError("the clock thread did not end")
        w.gpu.poll()


def gpu_convert(clocks, run_s=30.0, ready=True, clean=True, during=None, calls=None, pages=1, before_wake=None):
    """A fake of `engine.convert` on the route marker-server: the term of the attempt from the budget and the window of
    the request (as `pages.marker_server_route`), the event `server-wake` with the reservation, then `during(observer)`
    (a test moves the clocks or waits at a barrier), the events of progress, and a result with the bound of the cost."""
    def convert(source, out, request=None, limits=None, work=None, observer=None, name=None, model_root=None,
                datalab_key=None, server_endpoint=None, server_keys=None):
        ep, srv = server_endpoint, (request or {}).get("server") or {}
        out = Path(out)
        out.mkdir(parents=True)
        base = {"schema": "mdf-engine-result/1", "attempt": request.get("attempt"),
                "route": {"converter": "marker-server", "family": "pages", "chosen_by": "user", "reason": "imposed",
                          "steps": []}, "components": {}, "timings": {"total_seconds": run_s}, "limits": []}
        if calls is not None:
            calls.append({"request": request, "endpoint": ep.name if ep else None, "keys": sorted(server_keys or {})})

        def failed(code, cost=None, record=None):
            result = dict(base, status="failed", error={"code": code, "fatal": False}, inventory_complete=False)
            if cost is not None:
                result.update(cost=cost, marker_server=record)
            write_json(out, "engine.json", result)
            return result
        try:
            if ep is None or ep.name != srv.get("name"):
                raise EngineError(Code.SERVER_NOT_CONFIGURED, "endpoint")
            term = servers.term(ep, pages, budget_cents=srv.get("budget_cents"),
                                until=(srv.get("session") or {}).get("armed_until"), now=clocks.wall())
            if before_wake is not None:
                before_wake()
            stop = observer({"event": "step", "step": "server-wake",
                             "reserve": {"provider": ep.provider, "endpoint": ep.name, "identity": ep.identity_sha256(),
                                         "url": servers.canonical_url(ep.url), "reserve_cents": term["reserve_cents"],
                                         "rate": ep.rate_usd_per_s, "tail_idle_s": ep.tail_idle_s, "D": term["D"]}})
            if stop:
                raise EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, "server-wake")
        except EngineError as exc:
            return failed(exc.code.value)
        # the times of the attempt as the relay keeps them: the monotonic clock with an anchor at the wake, so a
        # suspend does not count in its bound (fact 10 of the plan r3)
        anchor = (clocks.wall(), clocks.mono())

        def epoch():
            return anchor[0] + (clocks.mono() - anchor[1])
        t_wake = anchor[0]
        if during is not None:
            during(observer)
        # the child of Marker as the engine announces it: an event the observer of the worker never throttles
        stop = observer({"event": "child"}) or observer({"event": "progress", "progress": {"stage": "marker"}})
        t_close = epoch()
        if not stop:
            t_close = max(t_close, t_wake + min(run_s, term["D"] - 5))
        t_last = t_close - 2 if ready else None
        ok = clean and not stop
        record = {"t_wake": t_wake, "t_last": t_last, "t_close": t_close}
        cost = {"provider": ep.provider, "bound_cents": servers.bound_cents(ep, ready, ok, t_wake, t_last, t_close),
                "clean": ok, "reserve_cents": term["reserve_cents"]}
        if stop:
            return failed("interrupted" if stop == "interrupted" else "cancelled", cost, record)
        write(out, "document.md", b"converted on a GPU")
        (out / "diagnostics").mkdir()
        write(out, "diagnostics/marker-1.log", b"log of the child")
        inventory = [identify(out, n) for n in ("document.md", "diagnostics/marker-1.log")]
        result = dict(base, status="complete", problems=[], problem_counts={}, problem_codes={}, inventory=inventory,
                      inventory_complete=True, cost=cost, marker_server=record)
        write_json(out, "engine.json", result)
        return result
    return convert
