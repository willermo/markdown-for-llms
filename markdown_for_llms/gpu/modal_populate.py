"""Fills the Modal Volume `mdf-surya-weights` with the pinned weights (MFL033; plan r4 §5.3 and C.1).

A separate CPU app (2 cores, 4 GiB, `timeout=1800`, no retries), run inside a `session.py
--cpu-activity` as `uv tool run --from modal==1.6.3 modal run scripts/servers/modal_populate.py::populate`.
It downloads the pinned revision into `/weights/hf` with the huggingface_hub of the vLLM image
(`watchdog.download` outside the proxy, killed after DOWNLOAD_LIMIT_S), checks the size and SHA-256
of model.safetensors (`watchdog.weights_ok`) and commits the Volume.
"""
from pathlib import Path

import modal

IMAGE = "vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268"
PYTHON_LINK = ["RUN ln -s /usr/bin/python3.12 /usr/local/bin/python"]   # as modal_app.py (MFL033 c7.10)
TIMEOUT_S = 1800
DOWNLOAD_LIMIT_S = 1700                               # inside the timeout, with time for the check and the commit
image = modal.Image.from_registry(IMAGE, setup_dockerfile_commands=PYTHON_LINK).entrypoint([]).add_local_file(
    str(Path(__file__).resolve().with_name("watchdog.py")), "/mdf/watchdog.py")
weights = modal.Volume.from_name("mdf-surya-weights", create_if_missing=True)
app = modal.App("mdf-surya-populate")


@app.function(image=image, cpu=2.0, memory=4096, timeout=TIMEOUT_S, retries=0, volumes={"/weights": weights},
              env={"HF_HOME": "/weights/hf", "HF_HUB_OFFLINE": "0"})
def populate():
    import json
    import sys
    sys.path.insert(0, "/mdf")
    import watchdog
    before = watchdog.weights_ok("/weights/hf")
    code = None
    if not before:
        code = watchdog.download(watchdog.Standalone(), "/weights/hf", DOWNLOAD_LIMIT_S)
    after = watchdog.weights_ok("/weights/hf")
    weights.commit()
    print(json.dumps({"populate": "mdf-surya-weights", "present_before": before, "download_exit": code,
                      "verified": after, "revision": watchdog.REVISION, "sha256": watchdog.WEIGHT[2]}))
    if not after:
        raise SystemExit(6)
