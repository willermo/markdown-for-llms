"""The Modal app of the marker-server route (MFL033; plan r2 §5.3, r3 §5.3, r4 §5.3 and C.1).

Deployed by `session.py` (the arm: `modal deploy`, with `MDF_MODAL_APP` = `mdf-surya` or
`mdf-surya-fault` and `MDF_FAULT` read at deploy time) and stopped by it (`modal app stop`). One
web function: the proxy of `watchdog.py` on port 8000 in front of vLLM, the weights read from the
Volume `mdf-surya-weights` (filled by `modal_populate.py`), at most one container, no retries, on
an L4 or else an A10G (c7.14). With the lever `compile-cache-populate` (c8.4) the Volume is committed
once the watchdog has written the cache of torch.compile on it.
Needs the SDK `modal==1.6.3`, outside the product.
"""
import os
from pathlib import Path

import modal

APP = os.environ.get("MDF_MODAL_APP", "mdf-surya")
IMAGE = "vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268"
# The vLLM image has Python 3.12 only as `python3` (update-alternatives, no `python`), while a
# registry image of a Modal Function must have `python` on PATH (modal 1.6.3, `Image.from_registry`);
# without it Modal refuses the Function (seen on 19/11 at the populate, MFL033 c7.10). The same line
# in modal_populate.py, so both apps share one image.
PYTHON_LINK = ["RUN ln -s /usr/bin/python3.12 /usr/local/bin/python"]
ENV = {"MDF_PROVIDER": "modal", "MDF_ALLOW_DOWNLOAD": "0", "PORT": "8000",
       "HF_HOME": "/weights/hf", "HF_HUB_OFFLINE": "1", "TMPDIR": "/tmp", "VLLM_CACHE_ROOT": "/tmp/vllm",
       "XDG_CACHE_HOME": "/tmp/cache"}
if os.environ.get("MDF_FAULT"):                      # no empty value (the lesson of RunPod, MFL033 c7.3)
    ENV["MDF_FAULT"] = os.environ["MDF_FAULT"]
if os.environ.get("MDF_LEVERS"):                     # the levers of the start of the session (MFL033 N5b)
    ENV["MDF_LEVERS"] = os.environ["MDF_LEVERS"]

image = modal.Image.from_registry(IMAGE, setup_dockerfile_commands=PYTHON_LINK).entrypoint([]).add_local_file(
    str(Path(__file__).with_name("watchdog.py")), "/mdf/watchdog.py")
# The populate (modal_populate.py) creates the Volume; this app never does: a deploy without it fails
# instead of serving from an empty Volume (r4 §5.3, MFL033 c7.8).
weights = modal.Volume.from_name("mdf-surya-weights", create_if_missing=False)
app = modal.App(APP)


# The GPUs in their order of preference (MFL033 c7.14): an L4, else an A10G, both of 24 GB, so the
# arguments of vLLM stay those of a 24 GB card. modal 1.6.3 takes a `list` as one ranked function per
# type (`_functions.py`: `isinstance(gpu, list)`, rank = position); a tuple is refused. The session's
# rate is the dearest of them (`providers.MODAL_GPUS` and `modal_rate`, the same list).
GPUS = ["L4", "A10G"]


# No `retries`: a web function has no retry policy without it, and modal 1.6.3 refuses one even with
# `retries=0` ("Web Functions do not support retries.", `_functions.py:789-791`; seen on 19/11 at the
# first deploy, MFL033 c7.12).
@app.function(image=image, gpu=GPUS, cpu=(2.0, 2.0), memory=(16384, 16384), timeout=600, startup_timeout=900,
              scaledown_window=60, min_containers=0, max_containers=1, volumes={"/weights": weights}, env=ENV)
@modal.concurrent(max_inputs=36)                     # 32 inferences + 4 for the probes and the metrics
@modal.web_server(port=8000, startup_timeout=900, requires_proxy_auth=True)
def serve():
    import sys
    import threading
    sys.path.insert(0, "/mdf")
    import watchdog
    if "compile-cache-populate" in os.environ.get("MDF_LEVERS", ""):      # MFL033 c8.4
        threading.Thread(target=commit_when_written, args=(watchdog.CACHE_WRITTEN,), daemon=True).start()
    watchdog.start_proxy()                            # no exit callback: the proxy never exits


def commit_when_written(marker, limit_s=1800, sleep=None):
    """c8.4: once the watchdog has written the archive of the cache of torch.compile under /weights (its
    path in `marker`), the Volume is committed, so that the next containers see it; one line
    `mdf_compile_cache`, `phase` commit, on stderr like the watchdog's."""
    import json
    import sys
    import time
    sleep = sleep or time.sleep
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit_s:
        if Path(marker).exists():
            try:
                weights.commit()
                ok, error = True, None
            except Exception as exc:
                ok, error = False, type(exc).__name__
            sys.stderr.write(json.dumps({"event": "mdf_compile_cache", "phase": "commit", "ok": ok, "error": error,
                                         "t": round(time.time(), 3)}, sort_keys=True) + "\n")
            sys.stderr.flush()
            return ok
        sleep(1)
    return False
