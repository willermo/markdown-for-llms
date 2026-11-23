"""The Modal app of the lever `snapshot` (MFL045 c2; plan r2 §4, §5, §7): the marker-server route with the memory
and GPU snapshot of Modal (an alpha feature of modal 1.6.3).

Deployed by `session.py` instead of `modal_app.py` when the arm names the lever `snapshot` (only on an app named
`*-snap` or `*-snap-fault`, with exactly `providers.SNAPSHOT_LEVERS`); stopped by it (`modal app stop`). The
image, the Volume `mdf-surya-weights` (never created here), the GPUs, the resources and the limits of
`modal_app.py`, in a class `Serve`, because Modal takes a snapshot only after the `enter(snap=True)` methods of a
class (a `web_server` function would be snapshotted before it serves, with the imports only):

- `enter(snap=True)` runs `snapshot.snap_boot` (`snapshot.py`, only in this image, next to `watchdog.py`): vLLM in
  sleep mode on the loopback, a warm-up with a synthetic page, the reset of the prefix cache, the sleep; before
  the proxy and before any input, so no document is ever in a snapshot. Any error raises: no snapshot.
- `enter(snap=False)` runs `snapshot.snap_wake` in every container, the one that created the snapshot included:
  the kind of the start, the proxy on port 8000 with the vLLM adopted, the wake, or the declared fallback to a
  normal start.
- the `web_server` method is empty: Modal waits for port 8000, which the proxy holds.

A deploy changed in anything (code, image, env: the levers, the fault) makes the snapshots of the app obsolete.
Needs the SDK `modal==1.6.3`, outside the product.
"""
import os
from pathlib import Path

import modal

APP = os.environ.get("MDF_MODAL_APP", "mdf-surya-snap")
IMAGE = "vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268"
PYTHON_LINK = ["RUN ln -s /usr/bin/python3.12 /usr/local/bin/python"]     # as modal_app.py (MFL033 c7.10)
ENV = {"MDF_PROVIDER": "modal", "MDF_ALLOW_DOWNLOAD": "0", "PORT": "8000",
       "HF_HOME": "/weights/hf", "HF_HUB_OFFLINE": "1", "TMPDIR": "/tmp", "VLLM_CACHE_ROOT": "/tmp/vllm",
       "XDG_CACHE_HOME": "/tmp/cache"}
if os.environ.get("MDF_FAULT"):                      # no empty value (MFL033 c7.3)
    ENV["MDF_FAULT"] = os.environ["MDF_FAULT"]
if os.environ.get("MDF_LEVERS"):
    ENV["MDF_LEVERS"] = os.environ["MDF_LEVERS"]

image = modal.Image.from_registry(IMAGE, setup_dockerfile_commands=PYTHON_LINK).entrypoint([]).add_local_file(
    str(Path(__file__).resolve().with_name("watchdog.py")), "/mdf/watchdog.py").add_local_file(
    str(Path(__file__).resolve().with_name("snapshot.py")), "/mdf/snapshot.py")
weights = modal.Volume.from_name("mdf-surya-weights", create_if_missing=False)
app = modal.App(APP)
GPUS = ["L4", "A10G"]                                # as modal_app.py: a list, ranked (MFL033 c7.14)


# No `retries`, as the web function of modal_app.py (MFL033 c7.12).
@app.cls(image=image, gpu=GPUS, cpu=(2.0, 2.0), memory=(16384, 16384), timeout=600, startup_timeout=900,
         scaledown_window=60, min_containers=0, max_containers=1, volumes={"/weights": weights}, env=ENV,
         enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True})
@modal.concurrent(max_inputs=36)                     # 32 inferences + 4 for the probes and the metrics
class Serve:
    @modal.enter(snap=True)
    def boot(self):
        import sys
        sys.path.insert(0, "/mdf")
        import snapshot
        self.snap = snapshot.snap_boot()

    @modal.enter(snap=False)
    def wake(self):
        import snapshot
        self.wd = snapshot.snap_wake(self.snap)

    @modal.web_server(port=8000, startup_timeout=900, requires_proxy_auth=True)
    def web(self):
        pass
