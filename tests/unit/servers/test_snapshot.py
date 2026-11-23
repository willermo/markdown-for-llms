"""The memory and GPU snapshot of Modal in the proxy (`scripts/servers/watchdog.py`, MFL045 c2; plan r2 §4,
§5, §7, App. D and E.7). `snap_boot` runs in `enter(snap=True)` of `modal_snap.py` before any input: vLLM in
sleep mode on the loopback, two chats with a synthetic page, the reset of the prefix cache, the sleep, then
`mdf_snapshot`; `snap_wake` runs in `enter(snap=False)` of every container, the one that created the
snapshot included: the kind of the start, the proxy with the vLLM adopted in its `waking` mode, the wake,
and the declared fallback to a normal start. No network: a fake vLLM as a function, a fake process, a fake
clock; the sockets of the proxy run in `watchdog_driver.py` and `snapshot_driver.py`."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import tarfile
import tempfile
import threading
import types
import unittest
import zlib
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]


def load():
    """`snapshot.py` (its `import watchdog` from scripts/servers): the patches go to the watchdog it uses, `sn.wd`."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "servers"))
    spec = importlib.util.spec_from_file_location("mdf_snapshot_t", ROOT / "scripts" / "servers" / "snapshot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sn = load()
wd = sn.wd
PAGE_SHA256 = "438a345895fcd6e4c137a0660cf6e9714ead96a97ffd88ef387fd871527b9785"
ARCHIVES = ("compile-cache:29012f5b05f56b51698c99f6d5f573d3511ae06191946be983554a8b35a55b2b,"
            "compile-cache:cd31ce211155dae8d34230a9988f7fd82e855c54b94dc4e79607a7bc93608224,"
            "compile-cache:d38180544831bf3c0d1a45712e25f125ff4d18a9929012376f844a2224e0ff73")


def events(text):
    return [json.loads(x) for x in text.splitlines() if x.startswith("{")]


class Clock:
    def __init__(self, t=5000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Proc:
    """The vLLM of `snap_boot` (a `watchdog.VLLM`): its leader pid, the signals sent to its group, its exit."""

    def __init__(self, pid=4242):
        self.proc = types.SimpleNamespace(pid=pid, wait=lambda timeout=None: 0)
        self.signals, self.code = [], None

    def signal(self, sig):
        self.signals.append(sig)

    def poll(self):
        return self.code


class FakeVLLM:
    """vLLM on the loopback, as `call(method, path, body, timeout)` sees it: ready after `ready_after` s of the
    fake clock; the answers of the routes of its dev mode overridable; every call recorded."""

    def __init__(self, clock, ready_after=40.0, **over):
        self.clock, self.t0, self.ready_after = clock, clock.t, ready_after
        self.calls, self.bodies, self.over = [], [], over
        self.asleep = False

    def __call__(self, method, path, body=None, timeout=None):
        self.calls.append((method, path))
        if body is not None:
            self.bodies.append(body)
        if path in self.over:
            answer = self.over[path]
            answer = answer() if callable(answer) else answer
            if isinstance(answer, BaseException):
                raise answer
            return answer
        if path in ("/health", "/v1/models") and self.clock.t - self.t0 < self.ready_after:
            return None, ""
        if path == "/sleep?level=1":
            self.asleep = True
        elif path == "/wake_up":
            self.asleep = False
        elif path == "/is_sleeping":
            return 200, json.dumps({"is_sleeping": self.asleep})
        return {"/health": (200, ""), "/v1/models": (200, json.dumps({"data": [{"id": wd.MODEL}]})),
                "/v1/chat/completions": (200, '{"choices": [{"message": {"content": "<p></p>"}}]}'),
                "/reset_prefix_cache": (200, ""), "/sleep?level=1": (200, ""),
                "/is_sleeping": (200, '{"is_sleeping": true}'), "/wake_up": (200, "")}.get(path, (404, "{}"))


class SnapCase(unittest.TestCase):
    """The weights of the Volume (a small fake model, its SHA-256 pinned for the test), the env of the deploy."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.saved = wd.WEIGHT
        self.addCleanup(setattr, wd, "WEIGHT", self.saved)
        data = b"weights-of-the-snapshot-test"
        wd.WEIGHT = ("model.safetensors", len(data), hashlib.sha256(data).hexdigest())
        self.hf = self.tmp / "vol" / "hf"
        snap = wd.snapshot(str(self.hf))
        snap.mkdir(parents=True)
        for name in wd.FILES:
            (snap / name).write_text("{}")
        (snap / "model.safetensors").write_bytes(data)
        self.env = {"MDF_PROVIDER": "modal", "HF_HOME": str(self.hf), "MDF_LEVERS": "snapshot," + ARCHIVES,
                    "MODAL_TASK_ID": "ta-01SNAPCREATE000000000000", "MODAL_ENABLE_SNAP_RESTORE": "1",
                    "MODAL_REGION": "us-east-1", "MODAL_CLOUD_PROVIDER": "CLOUD_PROVIDER_AWS",
                    "MODAL_IMAGE_ID": "im-01IMAGE", "MODAL_IDENTITY_TOKEN": "an-oidc-token-never-logged",
                    "PATH": "/usr/bin", "VLLM_CACHE_ROOT": str(self.tmp / "vllm")}
        self.clock = Clock()
        self.gpu = {"names": ["NVIDIA A10G"], "uuids": ["GPU-aaaa"], "source": "nvidia-smi", "error": None}
        self.spawned = []

    def spawn(self, argv, env):
        p = Proc()
        self.spawned.append((argv, env, p))
        return p

    def capture(self, fn):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                value = fn()
            except Exception as exc:                          # the events up to the exception, then the exception
                value = exc
        return value, events(err.getvalue())

    def boot(self, vllm=None, **kw):
        vllm = vllm or FakeVLLM(self.clock)
        with mock.patch.object(wd, "use_cache", side_effect=AssertionError("no use_cache in snap_boot")), \
                mock.patch.object(sn, "cpu_identity", lambda: {"model": "AMD EPYC 7R13", "flags_sha256": "f" * 16}), \
                mock.patch.object(sn, "group_pids", lambda pgid: [pgid, pgid + 40]):
            return self.capture(lambda: sn.snap_boot(env=self.env, call=vllm, clock=self.clock, wall=self.clock,
                                                     sleep=self.clock.sleep, gpu=dict(self.gpu), spawn=self.spawn,
                                                     **kw)) + (vllm,)


class TestSyntheticPage(unittest.TestCase):
    """plan r1 and r2 §5: a PNG of 896 x 896 made in the process with the standard library (bands, glyph-like
    blocks, a ruled table, no text, no file), its SHA-256 a constant: never a page of the set nor of a user.
    Its deflate stream is of stored blocks, so its bytes do not depend on the zlib of the container."""

    def test_a_valid_png_with_a_constant_sha256(self):
        page = sn.synthetic_page()
        self.assertEqual(page, sn.synthetic_page())
        self.assertEqual(hashlib.sha256(page).hexdigest(), PAGE_SHA256)
        self.assertEqual(sn.SYNTH_SHA256, PAGE_SHA256)
        self.assertEqual(page[:8], b"\x89PNG\r\n\x1a\n")
        length, kind = struct.unpack(">I4s", page[8:16])
        w, h, depth, color = struct.unpack(">IIBB", page[16:26])
        self.assertEqual((kind, w, h, depth, color), (b"IHDR", 896, 896, 8, 2))
        self.assertTrue(3136 <= w * h <= 6291456)                     # vLLM's min_pixels and max_pixels
        idat = page.index(b"IDAT")
        n = struct.unpack(">I", page[idat - 4:idat])[0]
        stream = page[idat + 4:idat + 4 + n]
        self.assertEqual(stream[:2], b"\x78\x01")
        self.assertEqual(stream[2] & 0b110, 0)                         # BTYPE 00: a stored block
        raw = zlib.decompress(stream)
        self.assertEqual(len(raw), h * (1 + w * 3))
        self.assertEqual({raw[i * (1 + w * 3)] for i in range(h)}, {0})
        self.assertGreater(raw.count(bytes([sn.SYNTH_INK])), 50_000)    # not a blank page

    def test_the_chat_of_the_warm_up(self):
        body = sn.synthetic_chat()
        url = body["messages"][0]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        import base64
        self.assertEqual(hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest(), PAGE_SHA256)
        self.assertEqual((body["model"], body["temperature"]), (wd.MODEL, 0))


class TestSnapBoot(SnapCase):
    def test_the_steps_in_order_and_the_snapshot_line(self):
        snap, ev, vllm = self.boot()
        self.assertIsInstance(snap, dict, snap)
        after = vllm.calls[[c[1] for c in vllm.calls].index("/v1/models"):]
        self.assertEqual(after, [("GET", "/v1/models"), ("POST", "/v1/chat/completions"),
                                 ("POST", "/v1/chat/completions"), ("POST", "/reset_prefix_cache"),
                                 ("POST", "/sleep?level=1"), ("GET", "/is_sleeping")])
        self.assertEqual({p for _, p in vllm.calls[:vllm.calls.index(("GET", "/v1/models"))]}, {"/health"})
        for body in vllm.bodies:
            self.assertEqual(body, sn.synthetic_chat())                 # the synthetic page only
        (argv, env, proc), = self.spawned
        self.assertEqual(argv[:3], ["vllm", "serve", wd.MODEL])
        self.assertIn("--enable-sleep-mode", argv)
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual((env["VLLM_SERVER_DEV_MODE"], env["TORCHINDUCTOR_COMPILE_THREADS"], env["HF_HUB_OFFLINE"]),
                         ("1", "1", "1"))
        self.assertNotIn("VLLM_SERVER_DEV_MODE", os.environ)            # only in the env of vLLM
        names = [e["event"] for e in ev]
        self.assertEqual(names, ["mdf_gpu", "mdf_weights_check", "mdf_vllm_args", "mdf_vllm_start", "mdf_snap_boot",
                                 "mdf_snapshot"])
        args = ev[2]
        self.assertTrue(args["snap_boot"])
        self.assertIn("--enable-sleep-mode", args["args"])
        self.assertEqual(ev[4]["ready_s"], 40.0)
        shot = ev[-1]
        self.assertTrue(shot["ok"])
        self.assertEqual(len(shot["gen"]), 16)
        self.assertEqual((shot["page_sha256"], shot["internal_chats"], shot["task"], shot["pids"]),
                         (PAGE_SHA256, 2, "ta-01SNAPCREATE000000000000", [4242, 4282]))
        self.assertEqual((shot["gpu"], shot["uuids"], shot["region"], shot["cloud"], shot["image"], shot["snap_enabled"]),
                         (["NVIDIA A10G"], ["GPU-aaaa"], "us-east-1", "CLOUD_PROVIDER_AWS", "im-01IMAGE", True))
        self.assertEqual(shot["cpu"], {"model": "AMD EPYC 7R13", "flags_sha256": "f" * 16})
        self.assertNotIn("an-oidc-token-never-logged", json.dumps(ev))
        self.assertEqual((snap["gen"], snap["task"], snap["vllm"]), (shot["gen"], shot["task"], proc))
        self.assertEqual(snap["cfg"]["levers"], wd.levers("snapshot," + ARCHIVES))
        self.assertEqual(proc.signals, [])

    def test_a_failure_kills_the_group_and_raises_so_no_snapshot(self):
        cases = (("not ready", dict(ready_after=10_000.0), "not_ready"),
                 ("a chat fails", {"/v1/chat/completions": (500, "{}")}, "chat"),
                 ("the reset fails", {"/reset_prefix_cache": (500, "")}, "reset_prefix_cache"),
                 ("the sleep fails", {"/sleep?level=1": (500, "")}, "sleep"),
                 ("still awake", {"/is_sleeping": (200, '{"is_sleeping": false}')}, "is_sleeping"),
                 ("a timeout", {"/sleep?level=1": socket.timeout("timed out")}, "sleep"))
        for name, over, reason in cases:
            with self.subTest(name):
                self.spawned.clear()
                ready_after = over.pop("ready_after", 40.0) if "ready_after" in over else 40.0
                value, ev, _ = self.boot(FakeVLLM(self.clock, ready_after=ready_after, **over))
                self.assertIsInstance(value, sn.SnapFailed)
                (_, _, proc), = self.spawned
                self.assertEqual(proc.signals, [signal.SIGKILL])
                self.assertEqual((ev[-1]["event"], ev[-1]["ok"]), ("mdf_snapshot", False))
                self.assertIn(reason, ev[-1]["reason"])

    def test_vllm_that_exits_before_readiness(self):
        def spawn(argv, env):
            p = Proc()
            p.code = 1
            self.spawned.append((argv, env, p))
            return p
        value, ev, _ = self.capture(lambda: sn.snap_boot(env=self.env, call=FakeVLLM(self.clock), clock=self.clock,
                                                         wall=self.clock, sleep=self.clock.sleep, gpu=dict(self.gpu),
                                                         spawn=spawn))[:2] + (None,)
        self.assertIsInstance(value, sn.SnapFailed)
        self.assertIn("vllm_exit", ev[-1]["reason"])

    def test_no_vllm_without_the_weights_or_with_another_lever(self):
        (self.hf.parent / "hf" / "hub").rename(self.tmp / "elsewhere")
        value, ev, _ = self.boot()
        self.assertIsInstance(value, sn.SnapFailed)
        self.assertEqual(self.spawned, [])
        self.assertEqual((ev[-1]["event"], ev[-1]["ok"], ev[-1]["reason"]), ("mdf_snapshot", False, "weights"))
        (self.tmp / "elsewhere").rename(self.hf / "hub")
        for levers in ("compile-cache-populate,snapshot", "nope", ARCHIVES):
            with self.subTest(levers=levers):
                self.env["MDF_LEVERS"] = levers
                value, ev, _ = self.boot()
                self.assertIsInstance(value, sn.SnapFailed)
                self.assertEqual(self.spawned, [])
                self.assertEqual(ev[-1]["reason"], "levers")

    def test_the_calls_go_only_to_the_loopback(self):
        hosts = []

        class Conn:
            def __init__(self, host, port, timeout=None):
                hosts.append((host, port))
                raise ConnectionRefusedError()
        with mock.patch.object(wd.http.client, "HTTPConnection", Conn):
            call = wd.upstream_call(8001)
            self.assertEqual(call("GET", "/health", timeout=1.0), (None, ""))
            self.assertEqual(call("POST", "/sleep?level=1", body={}, timeout=1.0), (None, ""))
        self.assertEqual(hosts, [("127.0.0.1", 8001)] * 2)


class TestLevers(unittest.TestCase):
    def test_the_lever_snapshot(self):
        self.assertIn("snapshot", wd.LEVERS)
        self.assertEqual(wd.levers("snapshot," + ARCHIVES)[-1], "snapshot")
        with self.assertRaises(wd.LeversInvalid):
            wd.levers("snapshot,compile-cache-populate")
        with self.assertRaises(wd.LeversInvalid):
            wd.levers("snapshot:1")


class WakeCase(SnapCase):
    def snap(self, **over):
        snap, ev, _ = self.boot()
        self.assertIsInstance(snap, dict)
        snap.update(over)
        return snap

    def wake(self, snap, env=None, gpu=None, vllm=None, thread=False):
        env = dict(self.env, **(env or {}))
        vllm = vllm or FakeVLLM(self.clock)
        w, ev = self.capture(lambda: sn.snap_wake(snap, env=env, call=vllm, clock=self.clock, wall=self.clock,
                                                  gpu=dict(gpu or self.gpu), serve=False, thread=thread))
        return w, ev, vllm


class TestSnapWake(WakeCase):
    def test_the_kind_of_the_start(self):
        """E.7: `create`, the same task as the snapshot (its container passes through the restore too);
        `restore`, another task; `boot`, snapshots not enabled for the task."""
        for env, enabled, kind in (({}, True, "create"), ({"MODAL_TASK_ID": "ta-01OTHER"}, True, "restore"),
                                   ({"MODAL_TASK_ID": "ta-01OTHER"}, False, "boot")):
            with self.subTest(kind=kind):
                snap = self.snap(snap_enabled=enabled)
                self.clock.t += 3600
                w, ev, vllm = self.wake(snap, env)
                restore = [e for e in ev if e["event"] == "mdf_restore"][0]
                self.assertEqual(restore["kind"], kind)
                self.assertEqual(restore["gen"], snap["gen"])
                self.assertEqual(restore["age_s"], 3600.0)
                self.assertEqual((restore["gpu_snap"], restore["gpu_now"]), (["NVIDIA A10G"], ["NVIDIA A10G"]))
                self.assertEqual(restore["pids"], [4242, 4282])
                start = [e for e in ev if e["event"] == "mdf_start"][0]
                self.assertEqual((start["snapshot"], start["boot_id"]), (kind, w.boot_id))
                self.assertEqual(w.restore_kind, kind)
                self.assertNotIn("an-oidc-token-never-logged", json.dumps(ev))

    def test_waking_then_awake(self):
        snap = self.snap()
        w, ev, vllm = self.wake(snap, thread=None)
        self.assertTrue(w.waking)
        self.assertEqual(w.refusal("/v1/models"), "mdf_waking")
        self.assertEqual(w.refusal("/v1/chat/completions"), "mdf_waking")
        w.probe = lambda path: (200, json.dumps({"data": [{"id": wd.MODEL}]}) if path == "/v1/models" else "")
        self.capture(w.tick)
        self.assertFalse(w.ready)                                       # /health of the proxy stays 204
        ok, ev2 = self.capture(lambda: sn.wake(w, snap, None, vllm, self.clock))
        self.assertTrue(ok)
        self.assertFalse(w.waking)
        self.assertIsNone(w.refusal("/v1/models"))
        self.assertEqual([c for c in vllm.calls], [("POST", "/wake_up"), ("GET", "/is_sleeping")])
        self.assertEqual([(e["event"], e["ok"]) for e in ev2], [("mdf_wake", True)])
        _, ev3 = self.capture(w.tick)
        self.assertTrue(w.ready)
        self.assertIn("mdf_ready", [e["event"] for e in ev3])
        self.assertIs(w.process, snap["vllm"])


class TestFallback(WakeCase):
    """P6, A10: the fallback is declared (`mdf_fallback` with its reason and the pids of the group), the group
    is killed, `torch_compile_cache` of the sleep-mode compilation is removed, and `prepare` starts vLLM as
    without the lever (no sleep mode, no dev mode) with the archive of the cache, within W1; the process is
    replaced under the lock, without `mdf_vllm_exit` of the one replaced."""

    def archive(self, gpu):
        """An archive of the cache for `gpu`, on the Volume, named by its SHA-256 (as the populate writes it)."""
        src = self.tmp / "src" / wd.CACHE_SUB / "g"
        src.mkdir(parents=True)
        (src / "graph.py").write_text("compiled without sleep mode\n")
        root = self.hf.parent / wd.CACHE_DIR
        root.mkdir(parents=True)
        tmp = root / "x.tar"
        manifest = json.dumps({"gpu": gpu, "levers": [], "files": 1, "bytes": 1}).encode()
        with tarfile.open(tmp, "w") as tar:
            info = tarfile.TarInfo(wd.CACHE_MANIFEST)
            info.size = len(manifest)
            tar.addfile(info, io.BytesIO(manifest))
            tar.add(str(src.parent), arcname=wd.CACHE_SUB)
        sha = wd.sha256_file(tmp)
        tmp.rename(root / ("torch-compile-%s.tar" % sha))
        return sha

    def fallback_case(self, reason_over=None, env=None, gpu=None, before=None):
        sha = self.archive(["NVIDIA A10G"])
        self.env["MDF_LEVERS"] = "snapshot,compile-cache:" + sha
        snap = self.snap()
        stale = Path(self.env["VLLM_CACHE_ROOT"]) / wd.CACHE_SUB / "sleep-mode"
        stale.mkdir(parents=True)
        (stale / "graph.py").write_text("compiled with sleep mode\n")
        if before:
            before(snap)
        started = []

        class Normal(Proc):
            def __init__(self, argv, env=None):
                super().__init__(pid=5151)
                started.append((argv, env))
        vllm = FakeVLLM(self.clock, **(reason_over or {}))
        with mock.patch.object(wd, "VLLM", Normal), mock.patch.dict(os.environ, {"VLLM_CACHE_ROOT":
                                                                                 self.env["VLLM_CACHE_ROOT"]}), \
                mock.patch.object(wd, "mtp_head", lambda h: (False, {})):
            w, ev, _ = self.wake(snap, env=env, gpu=gpu, vllm=vllm, thread=False)
        return w, ev, snap, started, vllm

    def check_fallback(self, w, ev, snap, started, reason, cache_used=True):
        names = [e["event"] for e in ev]
        fb = ev[names.index("mdf_fallback")]
        self.assertIn(reason, fb["reason"])
        self.assertEqual(fb["pids"], [4242, 4282])
        self.assertEqual(snap["vllm"].signals, [signal.SIGKILL])
        self.assertNotIn("mdf_vllm_exit", names)
        self.assertLess(names.index("mdf_fallback"), names.index("mdf_compile_cache"))
        cache = [e for e in ev if e["event"] == "mdf_compile_cache"][0]
        root = Path(self.env["VLLM_CACHE_ROOT"]) / wd.CACHE_SUB
        self.assertFalse((root / "sleep-mode").exists())
        if cache_used:                                                  # the archive, once the stale cache is gone
            self.assertTrue(cache["used"], cache)
            self.assertEqual((root / "g" / "graph.py").read_text(), "compiled without sleep mode\n")
        else:                                                           # the archive of another GPU: not used
            self.assertEqual((cache["used"], cache["tried"][0]["reason"]), (False, "another_gpu"))
        (argv, env), = started
        self.assertNotIn("--enable-sleep-mode", argv)
        self.assertNotIn("VLLM_SERVER_DEV_MODE", env)
        self.assertNotIn("TORCHINDUCTOR_COMPILE_THREADS", env)
        args = [e for e in ev if e["event"] == "mdf_vllm_args"][-1]
        self.assertNotIn("--enable-sleep-mode", args["args"])
        self.assertEqual(w.process.proc.pid, 5151)
        self.assertFalse(w.waking)
        snap["vllm"].code = -9                                          # the old group is dead: never its exit line
        w.probe = lambda path: (None, "")
        _, ev2 = self.capture(w.tick)
        self.assertNotIn("mdf_vllm_exit", [e["event"] for e in ev2])

    def test_a_wake_that_fails(self):
        for name, over, reason in (("500", {"/wake_up": (500, "{}")}, "wake_up 500"),
                                   ("timeout", {"/wake_up": socket.timeout("timed out")}, "timeout"),
                                   ("still asleep", {"/is_sleeping": (200, '{"is_sleeping": true}')}, "is_sleeping")):
            with self.subTest(name):
                self.setUp()
                w, ev, snap, started, _ = self.fallback_case(over, env={"MODAL_TASK_ID": "ta-01OTHER"})
                self.check_fallback(w, ev, snap, started, reason)

    def test_a_wake_over_its_limit(self):
        def slow():
            self.clock.t += sn.WAKE_S + 1
            return (200, "")
        w, ev, snap, started, _ = self.fallback_case({"/wake_up": slow}, env={"MODAL_TASK_ID": "ta-01OTHER"})
        self.check_fallback(w, ev, snap, started, "timeout")

    def test_another_gpu_name_but_not_another_uuid(self):
        w, ev, snap, started, vllm = self.fallback_case(env={"MODAL_TASK_ID": "ta-01OTHER"},
                                                        gpu={"names": ["NVIDIA L4"], "uuids": ["GPU-bbbb"],
                                                             "source": "nvidia-smi", "error": None})
        self.check_fallback(w, ev, snap, started, "another_gpu", cache_used=False)
        self.assertEqual(vllm.calls, [])                                # no wake on another GPU
        self.setUp()
        w, ev, snap, started, vllm = self.fallback_case(env={"MODAL_TASK_ID": "ta-01OTHER"},
                                                        gpu={"names": ["NVIDIA A10G"], "uuids": ["GPU-cccc"],
                                                             "source": "nvidia-smi", "error": None})
        self.assertNotIn("mdf_fallback", [e["event"] for e in ev])
        self.assertEqual(started, [])

    def test_wake_fail_only_on_a_restore(self):
        self.env["MDF_FAULT"] = "wake-fail"
        w, ev, snap, started, vllm = self.fallback_case(env={"MODAL_TASK_ID": "ta-01OTHER"})
        self.check_fallback(w, ev, snap, started, "wake-fail")
        self.assertEqual(vllm.calls, [])
        self.setUp()
        self.env["MDF_FAULT"] = "wake-fail"
        w, ev, snap, started, vllm = self.fallback_case()                 # the container that created it: no fault
        self.assertNotIn("mdf_fallback", [e["event"] for e in ev])
        self.assertEqual(vllm.calls, [("POST", "/wake_up"), ("GET", "/is_sleeping")])

    def test_a_vllm_that_died_during_the_snapshot(self):
        def dead(snap):
            snap["vllm"].code = 1
        w, ev, snap, started, vllm = self.fallback_case(env={"MODAL_TASK_ID": "ta-01OTHER"}, before=dead)
        self.check_fallback(w, ev, snap, started, "exit")

    def test_w1_on_a_wake_that_hangs(self):
        """A wake that never answers holds the proxy in `waking`: W1 trips 900 s after the start of the proxy,
        the group is killed and the proxy goes inert; no fallback after the trip."""
        release = threading.Event()

        def hang():
            release.wait(10)
            return (None, "")
        snap = self.snap()
        vllm = FakeVLLM(self.clock, **{"/wake_up": hang})
        w, ev = self.capture(lambda: sn.snap_wake(snap, env=dict(self.env, MODAL_TASK_ID="ta-01OTHER"), call=vllm,
                                                  clock=self.clock, wall=self.clock, gpu=dict(self.gpu), serve=False,
                                                  thread=True))
        w.probe = lambda path: (200, json.dumps({"data": [{"id": wd.MODEL}]}) if path == "/v1/models" else "")
        rule = None
        for _ in range(181):
            self.clock.t += 5
            rule, _ = self.capture(w.tick)
            if rule:
                break
        self.assertEqual(rule, "W1")
        self.assertEqual(w.inert, "mdf_inert")
        self.assertIn(signal.SIGKILL, snap["vllm"].signals)
        prepared = []
        with mock.patch.object(wd, "prepare", lambda *a, **k: prepared.append(a)):
            release.set()
            for t in threading.enumerate():
                if t.name == "mdf-wake":
                    t.join(10)
        self.assertEqual(prepared, [])                                 # no fallback after the trip


if __name__ == "__main__":
    unittest.main()
