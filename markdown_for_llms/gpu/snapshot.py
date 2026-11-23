"""The memory and GPU snapshot of Modal for the proxy (MFL045 c2; plan r2 §4, §5, §7, E.7), the lever `snapshot`.

Only in the image of `modal_snap.py`, next to `watchdog.py` (`/mdf`): RunPod receives the watchdog base64 in the env
of its endpoint, within a limit of 102,400 bytes for a request (HTTP 413 on 22/11, F0 of MFL045), and needs none of
this. Standard library and the watchdog only.

The lever `snapshot` (MFL045 c2; plan r2 §4, §5, §7, E.7) is the memory and GPU snapshot of Modal, on the
app of `modal_snap.py`: `snap_boot` runs in `enter(snap=True)`, before the proxy and before any input, and
starts vLLM on the loopback with `--enable-sleep-mode` (`VLLM_SERVER_DEV_MODE=1` and
`TORCHINDUCTOR_COMPILE_THREADS=1` only in its env, no archive of the cache), waits for it within
SNAP_BOOT_S, sends two chats with a synthetic page made here (never a page of the set nor of a user),
resets the prefix cache, puts vLLM to sleep at level 1 (the KV cache and the multimodal cache dropped) and
writes `mdf_snapshot` with the generation (a random `gen`, `t_snap`) and the identities of the container
(task, GPUs with their UUIDs, CPU, region, cloud, image); any error kills the group of vLLM and raises, so
no snapshot is taken. `snap_wake` runs in `enter(snap=False)` of every container, the one that created
the snapshot included (Modal restores it too): `mdf_restore` with the kind of the start (`create`, the same
task; `restore`, another task; `boot`, no snapshot for the task), then the proxy with the vLLM adopted in
its `waking` mode (503 `mdf_waking` to all but the health checks, /health 204, W1-W5 from its start), and a
thread that wakes vLLM within WAKE_S (`mdf_wake`). A wake that fails or takes longer, another name of GPU
(another UUID is normal) or the fault `wake-fail` on a `restore` is the declared fallback (`mdf_fallback`
with its reason and the pids of the old group): the group is killed, `torch_compile_cache` of the sleep
mode removed, and `prepare` starts vLLM as without the lever, with the archive of the cache, within W1.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import struct
import threading
import time
import zlib

import watchdog as wd

SNAP_BOOT_S = 600               # vLLM ready within so long in `snap_boot`: a creation stays within `ready_within`
SNAP_POLL_S = 2.0
SNAP_PROBE_S = 5.0              # each /health and /v1/models of the readiness of `snap_boot`
SNAP_CHAT_S = 120.0             # each chat of the warm-up
SNAP_CALL_S = 60.0              # the reset, the sleep, /is_sleeping
SNAP_SLEEP = "/sleep?level=1"   # level 1: the weights to the CPU, the KV cache and the multimodal cache dropped
SNAP_CHATS = 2
WAKE_S = 120.0                  # the wake (/wake_up, /is_sleeping) within so long, else the fallback
GROUP_PIDS = 64
# The synthetic page of the warm-up (plan r1 and r2 §5): a PNG of 896 x 896, RGB, made here with the standard
# library: a title band, lines of glyph-like blocks, a ruled table; no text, no file; its deflate stream of stored
# blocks, so its bytes do not depend on the zlib of the container. Never a page of the set nor of a user.
SYNTH_SIDE = 896
SYNTH_INK, SYNTH_PAPER = 32, 250
SYNTH_SHA256 = "438a345895fcd6e4c137a0660cf6e9714ead96a97ffd88ef387fd871527b9785"
SYNTH_PROMPT = "OCR this image to HTML."
SYNTH_MAX_TOKENS = 256


class SnapFailed(Exception):
    """`snap_boot` failed: the group of vLLM is killed and `mdf_snapshot` says `ok` false; no snapshot is taken."""


def _spans(y):
    """The dark spans of row `y` of the synthetic page: a title band, 14 lines of glyphs, a ruled table."""
    w, out = SYNTH_SIDE, []
    if 56 <= y < 80:
        x = 96
        for j in range(9):
            g = 22 + (j * 11) % 17
            out.append((x, x + g))
            x += g + (14 if j % 3 == 2 else 6)
    line, dy = divmod(y - 120, 34)
    if 0 <= line < 14 and dy < 13:
        x, j = 64, 0
        while True:
            g = 5 + (line * 13 + j * 7) % 9
            if x + g > w - 64:
                break
            out.append((x, x + g))
            x += g + (11 if (j + line) % (3 + (line + j) % 4) == 0 else 3)
            j += 1
    if 620 <= y <= 836:
        if (y - 620) % 36 < 2:
            out.append((64, w - 64))
        else:
            for x in (64, 300, 560, w - 66):
                out.append((x, x + 2))
            if 10 <= (y - 620) % 36 < 22:
                row = (y - 620) // 36
                for x0, x1 in ((80, 280), (316, 540), (576, w - 80)):
                    n = 3 + (row * 5 + x0) % 4
                    step = (x1 - x0) // (n + 1)
                    for k in range(n):
                        a = x0 + k * step
                        out.append((a, a + step - 6))
    return out


def synthetic_page():
    """The PNG of the warm-up (SYNTH_SHA256), made in memory."""
    w = h = SYNTH_SIDE
    raw = bytearray()
    for y in range(h):
        row = bytearray([SYNTH_PAPER]) * (w * 3)
        for a, b in _spans(y):
            row[a * 3:b * 3] = bytes([SYNTH_INK]) * ((b - a) * 3)
        raw += b"\x00" + row
    data = bytes(raw)
    z = bytearray(b"\x78\x01")
    for i in range(0, len(data), 65535):
        block = data[i:i + 65535]
        z += struct.pack("<BHH", 1 if i + 65535 >= len(data) else 0, len(block), 0xFFFF ^ len(block)) + block
    z += struct.pack(">I", zlib.adler32(data) & 0xFFFFFFFF)

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", bytes(z)) + chunk(b"IEND", b""))


def synthetic_chat():
    """The chat of the warm-up: the synthetic page and the prompt of Surya's pages."""
    data = base64.b64encode(synthetic_page()).decode("ascii")
    return {"model": wd.MODEL, "max_tokens": SYNTH_MAX_TOKENS, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
        {"type": "text", "text": SYNTH_PROMPT}]}]}


def identity(env=None):
    """The identity of a Modal container (plan r2 E.2), from named variables only: never MODAL_IDENTITY_TOKEN, an
    OIDC token. After a restore Modal's runtime rewrites them (`memory_restore`, `config.override_locally`)."""
    env = os.environ if env is None else env
    return {"task": env.get("MODAL_TASK_ID"), "region": env.get("MODAL_REGION"),
            "cloud": env.get("MODAL_CLOUD_PROVIDER"), "image": env.get("MODAL_IMAGE_ID"),
            "snap_enabled": env.get("MODAL_ENABLE_SNAP_RESTORE") == "1"}


def cpu_identity(path="/proc/cpuinfo"):
    """The model of the CPU and the SHA-256 (16 hex digits) of its flags: Modal ties a snapshot to the type of its
    worker by them (the guide of the memory snapshots, 22/11); an inference, not an identity of the platform."""
    model = flags = None
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "model name" and model is None:
                model = value.strip()[:wd.RUNTIME_TEXT]
            elif key == "flags" and flags is None:
                flags = " ".join(sorted(set(value.split())))
            if model is not None and flags is not None:
                break
    except OSError:
        pass
    return {"model": model, "flags_sha256": None if flags is None else hashlib.sha256(flags.encode()).hexdigest()[:16]}


def group_pids(pgid, root="/proc"):
    """The pids of the process group `pgid` (the group of vLLM: its API server and its engine), from /proc."""
    out = []
    try:
        for d in Path(root).iterdir():
            if not d.name.isdigit():
                continue
            try:
                fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if len(fields) > 2 and fields[2] == str(pgid):
                out.append(int(d.name))
    except OSError:
        pass
    return sorted(out)[:GROUP_PIDS]


def _object(text):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _step(call, method, path, name, timeout, body=None):
    """One step of `snap_boot` that must answer 200: its text, else SnapFailed with the name of the step."""
    try:
        status, text = call(method, path, body, timeout)
    except Exception as exc:
        raise SnapFailed("%s: %s" % (name, type(exc).__name__)) from None
    if status != 200:
        raise SnapFailed("%s %s" % (name, status))
    return text


def snap_boot(env=None, argv=None, call=None, clock=time.monotonic, wall=time.time, sleep=time.sleep, gpu=None,
              spawn=None):
    """`enter(snap=True)` of modal_snap.py (MFL045 c2; module docstring): vLLM ready in sleep mode, the warm-up with
    the synthetic page, the reset, the sleep, then `mdf_snapshot`. Returns what the snapshot keeps in memory: the
    vLLM process (`vllm`), the config (`cfg`), the generation and the identities at the snapshot. SnapFailed on
    any error, after the group of vLLM is killed; nothing here ever reaches beyond the loopback."""
    env = os.environ if env is None else env
    cfg = wd.config(env)
    call = call or wd.upstream_call(cfg["upstream"])
    spawn = spawn or (lambda a, e: wd.VLLM(a, env=e))
    st = wd.Standalone(clock)
    t0, proc = clock(), None
    gpu = wd.gpu_names() if gpu is None else gpu
    cfg["gpu_names"] = gpu["names"]
    wd.log("mdf_gpu", **gpu, since_start=st.since_start(), snap_boot=True)
    try:
        if not wd.check_weights(st, cfg["hf_home"], "snap_boot"):
            raise SnapFailed("weights")
        if cfg.get("levers_error") or "snapshot" not in (cfg.get("levers") or ()):
            raise SnapFailed("levers")
        try:
            mtp, fields = wd.mtp_head(cfg["hf_home"])
        except wd.ConfigInvalid as exc:
            raise SnapFailed("config %s" % exc) from None
        argv = argv or wd.vllm_argv(cfg, mtp) + ["--enable-sleep-mode"]
        venv = wd.vllm_env(env, HF_HOME=cfg["hf_home"], HF_HUB_OFFLINE="1", VLLM_SERVER_DEV_MODE="1",
                           TORCHINDUCTOR_COMPILE_THREADS="1")
        wd.log("mdf_vllm_args", ok=True, snap_boot=True, mtp=mtp, config=fields,
               args=[a for a in argv if a.startswith("--")], levers=list(cfg["levers"]), **wd.runtime(argv[0], venv),
               since_start=st.since_start())
        proc = spawn(argv, venv)
        wd.log("mdf_vllm_start", pid=proc.proc.pid, snap_boot=True, since_start=st.since_start())
        t_spawn = clock()
        while True:
            code = proc.poll()
            if code is not None:
                raise SnapFailed("vllm_exit %s" % code)
            if clock() - t_spawn >= SNAP_BOOT_S:
                raise SnapFailed("not_ready within %d s" % SNAP_BOOT_S)
            status, _ = call("GET", "/health", None, SNAP_PROBE_S)
            if status == 200:
                ms, text = call("GET", "/v1/models", None, SNAP_PROBE_S)
                if ms == 200 and wd.MODEL in [m.get("id") for m in _object(text).get("data") or [] if isinstance(m, dict)]:
                    break
            sleep(SNAP_POLL_S)
        ready_s = round(clock() - t_spawn, 1)
        wd.log("mdf_snap_boot", ready_s=ready_s, since_start=st.since_start())
        chat = synthetic_chat()
        for _ in range(SNAP_CHATS):
            _step(call, "POST", "/v1/chat/completions", "chat", SNAP_CHAT_S, body=chat)
        _step(call, "POST", "/reset_prefix_cache", "reset_prefix_cache", SNAP_CALL_S)
        _step(call, "POST", SNAP_SLEEP, "sleep", SNAP_CALL_S)
        text = _step(call, "GET", "/is_sleeping", "is_sleeping", SNAP_CALL_S)
        if _object(text).get("is_sleeping") is not True:
            raise SnapFailed("is_sleeping %s" % text[:40])
        ident = identity(env)
        snap = {"gen": secrets.token_hex(8), "t_snap": round(wall(), 3), "mono_snap": round(clock(), 3),
                "task": ident["task"], "gpu": gpu["names"], "uuids": gpu.get("uuids"), "cpu": cpu_identity(),
                "region": ident["region"], "cloud": ident["cloud"], "image": ident["image"],
                "snap_enabled": ident["snap_enabled"], "pids": group_pids(proc.proc.pid), "internal_chats": SNAP_CHATS,
                "page_sha256": SYNTH_SHA256, "ready_s": ready_s, "seconds": round(clock() - t0, 1)}
        wd.log("mdf_snapshot", ok=True, since_start=st.since_start(), **snap)
        snap.update(vllm=proc, cfg=cfg)
        return snap
    except Exception as exc:
        reason = exc.args[0] if isinstance(exc, SnapFailed) else "%s: %s" % (type(exc).__name__, str(exc)[:200])
        if proc is not None:
            proc.signal(signal.SIGKILL)
        wd.log("mdf_snapshot", ok=False, reason=str(reason)[:300], seconds=round(clock() - t0, 1),
               since_start=st.since_start())
        if isinstance(exc, SnapFailed):
            raise
        raise SnapFailed(reason) from None


def snap_wake(snap, env=None, call=None, clock=time.monotonic, wall=time.time, gpu=None, serve=True, thread=True):
    """`enter(snap=False)` of modal_snap.py (MFL045 c2; module docstring): the kind of the start in `mdf_restore`,
    the proxy with the vLLM adopted in its `waking` mode (W1-W5 from its start), then the wake in a thread
    (`thread` True; False runs it here, None not at all: the tests). Returns the watchdog."""
    env = os.environ if env is None else env
    cfg = snap["cfg"]
    call = call or wd.upstream_call(cfg["upstream"])
    ident = identity(env)
    now_gpu = wd.gpu_names() if gpu is None else gpu
    kind = "boot" if not snap["snap_enabled"] else ("create" if ident["task"] == snap["task"] else "restore")
    wd.log("mdf_restore", kind=kind, gen=snap["gen"], t_snap=snap["t_snap"], age_s=round(wall() - snap["t_snap"], 1),
           mono_jump_s=round(clock() - snap["mono_snap"], 1), task_snap=snap["task"], task_now=ident["task"],
           gpu_snap=snap["gpu"], gpu_now=now_gpu["names"], uuid_snap=snap["uuids"], uuid_now=now_gpu.get("uuids"),
           cpu_snap=snap["cpu"], cpu_now=cpu_identity(), region_snap=snap["region"], region_now=ident["region"],
           cloud_snap=snap["cloud"], cloud_now=ident["cloud"], image_snap=snap["image"], image_now=ident["image"],
           snap_enabled_now=ident["snap_enabled"], pids=snap["pids"])
    w = wd.Watchdog(cfg, clock=clock, process=snap["vllm"])
    w.waking, w.restore_kind = True, kind
    if serve:
        w.serve()
    wd.start_line(w, cfg, snapshot=kind)
    if serve:
        w.loops()
    cfg["gpu_names"] = now_gpu["names"]                       # the GPU of the archive of a fallback
    wd.log("mdf_gpu", **now_gpu, since_start=w.since_start())
    reason = None
    if now_gpu["names"] != snap["gpu"]:
        reason = "another_gpu: %s, the snapshot on %s" % (now_gpu["names"], snap["gpu"])
    elif cfg.get("fault") == "wake-fail" and kind == "restore":
        reason = "wake-fail"
    if thread is True:
        threading.Thread(target=wake, args=(w, snap, reason, call, clock), name="mdf-wake", daemon=True).start()
    elif thread is False:
        wake(w, snap, reason, call, clock)
    return w


def wake(w, snap, reason=None, call=None, clock=time.monotonic):
    """The wake of the adopted vLLM within WAKE_S (`/wake_up`, then `/is_sleeping` false): `mdf_wake` and the
    waking mode off; else, or with a `reason` already, the fallback. `w` is the watchdog. True when awake."""
    call = call or wd.upstream_call(w.cfg["upstream"])
    t0 = clock()
    if reason is None:
        try:
            code = snap["vllm"].poll()
            if code is not None:
                reason = "exit %s" % code
            else:
                status, _ = call("POST", "/wake_up", None, WAKE_S)
                if clock() - t0 >= WAKE_S:
                    reason = "timeout: the wake took %.0f s" % (clock() - t0)
                elif status != 200:
                    reason = "wake_up %s" % status
                else:
                    status, text = call("GET", "/is_sleeping", None, max(1.0, WAKE_S - (clock() - t0)))
                    if clock() - t0 >= WAKE_S:
                        reason = "timeout: the wake took %.0f s" % (clock() - t0)
                    elif status != 200 or _object(text).get("is_sleeping") is not False:
                        reason = "is_sleeping %s %s" % (status, text[:40])
        except Exception as exc:
            reason = "timeout" if isinstance(exc, socket.timeout) else "%s: %s" % (type(exc).__name__, str(exc)[:100])
    if reason is None:
        with w.lock:
            w.waking = False
        wd.log("mdf_wake", ok=True, seconds=round(clock() - t0, 1), since_start=w.since_start())
        return True
    fallback(w, snap, reason)
    return False


def fallback(w, snap, reason):
    """The declared fallback (P6, A10): `mdf_fallback`, the group of the adopted vLLM killed (it was taken off the
    proxy under its lock first, so its exit is never `mdf_vllm_exit`), `torch_compile_cache` of the sleep mode
    removed, then `prepare` as without the lever, with the archive of the cache, within W1; nothing after a trip.
    `w` is the watchdog."""
    old = snap["vllm"]
    with w.lock:
        tripped = w.inert is not None
        if w.process is old:
            w.process = None
    wd.log("mdf_fallback", reason=str(reason)[:300], kind=w.restore_kind, pids=snap["pids"], tripped=tripped,
           since_start=w.since_start())
    old.signal(signal.SIGKILL)
    try:
        old.proc.wait(timeout=30)
    except Exception:
        pass
    if tripped:
        return
    shutil.rmtree(wd.compile_root() / wd.CACHE_SUB, ignore_errors=True)
    w.seen = {"health": False, "models": False, "models_200": False, "exit": False}   # those of the new vLLM
    try:
        wd.prepare(w, w.cfg)
    except Exception as exc:
        wd.log("mdf_error", where="fallback", error=type(exc).__name__, detail=str(exc)[:300])
        w.go_inert("mdf_start_error")
    finally:
        with w.lock:
            w.waking = False
