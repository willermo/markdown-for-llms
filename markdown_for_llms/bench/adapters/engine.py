"""The conversion engine as `mdf bench run` sees it: one conversion, and the identity of what converts.

The bench converts only through `convert`, which is `engine.core.convert` (plan r2 §7): the route,
the options and the attempt ID in the request, the execution caps frozen in the state of the run,
the work directory of the run. `identity` is what a resumed run must find unchanged: the package
(its version and the SHA-256 of its installed `RECORD`), the toolchain image and the weights of the
Marker route. Tests replace these functions here, never inside the engine.
"""
import hashlib

from ...engine.codes import EngineError


def convert(source, out, request, limits, work, observer=None, name=None, model_root=None, datalab_key=None,
            server_endpoint=None, server_keys=None):
    from ...engine import core
    return core.convert(source, out, request, limits, work, observer=observer, name=name, model_root=model_root,
                        datalab_key=datalab_key, server_endpoint=server_endpoint, server_keys=server_keys)


def package_identity():
    """{version, record_sha256} of the installed distribution (None when it is not installed)."""
    from importlib.metadata import PackageNotFoundError, distribution
    try:
        dist = distribution("markdown-for-llms")
    except PackageNotFoundError:
        return {"version": None, "record_sha256": None}
    record = dist.read_text("RECORD")
    return {"version": dist.version,
            "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest() if record is not None else None}


def identity(model_root):
    """{package, toolchain, weights}: a component that cannot be verified carries its closed code."""
    from ...engine import toolchain, weights
    tc = {"content_sha256": toolchain.identity(), "image_id": None, "engine": None, "code": None}
    try:
        tc["image_id"] = toolchain.image()
        tc["engine"] = (toolchain.report().get("engine") or {}).get("version")
    except EngineError as exc:
        tc["code"] = exc.code.value
    try:
        w = weights.verify(model_root)
        wid = {"manifest_sha256": w["manifest_sha256"], "files": w["files"], "bytes": w["bytes"], "code": None}
    except EngineError as exc:
        wid = {"manifest_sha256": weights.manifest_sha256(), "files": None, "bytes": None, "code": exc.code.value}
    return {"package": package_identity(), "toolchain": tc, "weights": wid}
