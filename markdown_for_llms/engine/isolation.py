"""The light child of the engine: a Python module of the engine run in Bubblewrap.

Used for the steps that read untrusted bytes in Python (the package audit, `pkgaudit`), so that
a ZIP bomb, a deep XML tree or a parser defect stays inside a process without network, with a
read-only system and only its scratch directory writable:

    bwrap --unshare-all --die-with-parent --new-session --cap-drop ALL --clearenv
          --ro-bind /usr /lib /lib64 <base prefix> <prefix>   (the interpreter, read-only)
          --ro-bind <markdown_for_llms> /mdf/markdown_for_llms (the engine code, the same as the caller's)
          --ro-bind <input> /in/<name> ...                     (inputs, each alone, read-only)
          --bind <scratch> /scratch                            (the only writable path)
          --remount-ro /

`child.py` sets the resource limits (address space, file size, CPU time, processes, files)
before it imports the module. The caller bounds the wall time, the size of the scratch
directory and the output it reads; a missing Bubblewrap is `isolation_unavailable`, never a
fallback to an unisolated run. The product's `sandbox.py` (the Marker child of today) is left
as it is until tranche B of MFL032 moves the product onto the engine.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .codes import Code, EngineError

BWRAP = "/usr/bin/bwrap"
PACKAGE = Path(__file__).resolve().parents[1]          # the markdown_for_llms directory of this code
BOOT = ("import sys; sys.path.insert(0, '/mdf'); from markdown_for_llms.engine.child import main; "
        "raise SystemExit(main())")
LOG_LIMIT = 1 << 20


def available():
    return os.path.isfile(BWRAP) and os.access(BWRAP, os.X_OK)


def command(scratch, module, args=(), inputs=()):
    """The bwrap argv of a child running `module.main(args)` with `inputs` [(host path, name)]."""
    argv = [BWRAP, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv"]
    for path in ("/usr", "/lib", "/lib64", sys.base_prefix, sys.prefix):
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    argv += ["--proc", "/proc", "--dev", "/dev", "--dir", "/in", "--dir", "/mdf",
             "--ro-bind", str(PACKAGE), "/mdf/markdown_for_llms"]
    for host, name in inputs:
        if "/" in name or name in ("", ".", ".."):
            raise ValueError("input name: " + name)
        argv += ["--ro-bind", os.path.realpath(host), "/in/" + name]
    argv += ["--bind", str(Path(scratch).absolute()), "/scratch", "--chdir", "/scratch", "--remount-ro", "/"]
    for key, value in {"PATH": "/usr/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": "/scratch",
                       "TMPDIR": "/scratch"}.items():
        argv += ["--setenv", key, value]
    return argv + ["--", sys.executable, "-I", "-B", "-c", BOOT, "limits.json", module, *map(str, args)]


def scratch_bytes(path):
    total = 0
    for parent, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                info = os.lstat(os.path.join(parent, name))
            except OSError:
                continue
            total += max(info.st_size, info.st_blocks * 512)
    return total


def run(scratch, module, args=(), inputs=(), memory_mib=1024, output_mib=512, timeout=120, cpu_seconds=None,
        steps=None, observer=None, poll_seconds=0.1):
    """Run the child; returns its exit status and log (stdout and stderr, at most `LOG_LIMIT`
    bytes). `output_limit` when the scratch grows over `output_mib`, `timeout` past `timeout`
    seconds, `memory_limit` when the child reports it, `cancelled`/`interrupted` from the observer.
    The child writes its results into the scratch directory."""
    if not available():
        raise EngineError(Code.ISOLATION_UNAVAILABLE, "bubblewrap not found")
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    limits = {"memory_mib": memory_mib, "output_mib": output_mib,
              "cpu_seconds": cpu_seconds or max(1, int(timeout)), "process_limit": 16}
    (scratch / "limits.json").write_text(json.dumps(limits))
    argv = command(scratch, module, args, inputs)
    started = time.monotonic()
    reason = None
    log_path = scratch.parent / (scratch.name + ".log")
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env={},
                                close_fds=True, start_new_session=True)
        try:
            if observer is not None and (stop := observer({"event": "child", "module": module, "pid": proc.pid})):
                reason = EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, module)
            def over():
                if scratch_bytes(scratch) > output_mib * 1048576:
                    return EngineError(Code.OUTPUT_LIMIT, f"{module}: scratch over {output_mib} MiB")
                if os.fstat(log.fileno()).st_size > LOG_LIMIT:
                    return EngineError(Code.OUTPUT_LIMIT, f"{module}: log over {LOG_LIMIT} bytes")
                return None
            while reason is None:
                try:
                    proc.wait(timeout=poll_seconds)
                    reason = over()             # a child may fill its scratch and exit between two polls
                    break
                except subprocess.TimeoutExpired:
                    pass
                reason = over()
                if reason is None and time.monotonic() - started > timeout:
                    reason = EngineError(Code.TIMEOUT, f"{module}: timeout")
                if reason is None and observer is not None and (
                        stop := observer({"event": "tick", "module": module, "pid": proc.pid})):
                    reason = EngineError(Code.INTERRUPTED if stop == "interrupted" else Code.CANCELLED, module)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    data = log_path.read_bytes()[-LOG_LIMIT:]
    log_path.unlink()
    if steps is not None:
        steps.append({"tool": "bwrap:" + module.rsplit(".", 1)[-1], "args": [str(a) for a in args],
                      "seconds": round(time.monotonic() - started, 3), "exit": proc.returncode})
    if reason is not None:
        reason.log = data
        raise reason
    return proc.returncode, data


def clean(scratch):
    shutil.rmtree(scratch, ignore_errors=True)
