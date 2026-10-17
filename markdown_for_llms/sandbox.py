"""Bubblewrap child execution, finite limits, cancellation and bounded logs."""
from pathlib import Path
import json
import os
import selectors
import signal
import subprocess
import sys
import time

from .files import directory, write_json


def command(settings, scratch, source=None, module="markdown_for_llms.engines.marker", args=(), label="process"):
    scratch = Path(scratch).absolute()
    with directory(scratch):
        pass
    args_out = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                "--cap-drop", "ALL", "--clearenv"]
    for path in ("/usr", "/lib", "/lib64", sys.base_prefix, sys.prefix):
        if Path(path).exists():
            args_out += ["--ro-bind", path, path]
    args_out += ["--proc", "/proc", "--dev", "/dev", "--dir", "/etc",
                 "--ro-bind-try", "/etc/fonts", "/etc/fonts",
                 "--bind", str(scratch), "/scratch",
                 "--symlink", "/scratch/tmp", "/tmp", "--chdir", "/scratch"]
    (scratch / "tmp").mkdir(exist_ok=True)
    (scratch / "home").mkdir(exist_ok=True)
    if source:
        args_out += ["--ro-bind", str(source), "/source.pdf"]
    if module.endswith(".marker"):
        with directory(settings.model_root):
            pass
        args_out += ["--ro-bind", settings.model_root, "/models"]
    # The only writable host mount is the attempt scratch. Private /dev and
    # /proc are kernel interfaces, not host sockets or host process views.
    args_out += ["--remount-ro", "/"]
    env = {"PATH": "/usr/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "HOME": "/scratch/home", "TMPDIR": "/scratch/tmp", "XDG_CACHE_HOME": "/scratch/tmp",
           "TORCH_DEVICE": "cpu", "MODEL_CACHE_DIR": "/models", "FONT_DIR": "/models/fonts",
           "FONT_PATH": "/models/fonts/GoNotoCurrent-Regular.ttf",
           "RECOGNITION_RENDER_FONTS": json.dumps({k: "/models/fonts/" + (
               "GoNotoCurrent-Regular.ttf" if k == "all" else "GoNotoCJKCore.ttf") for k in ("all", "zh", "ja", "ko")}),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "OMP_NUM_THREADS": str(settings.threads), "MKL_NUM_THREADS": str(settings.threads)}
    for key, value in env.items():
        args_out += ["--setenv", key, value]
    return args_out + ["--", sys.executable, "-I", "-B", "-m",
                       "markdown_for_llms.child", label + "-limits.json", module, *map(str, args)]


def process_identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start": raw[19], "state": raw[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def tree_usage(pid):
    """Sample owned descendants; this is not a hard aggregate memory limit."""
    todo, seen, rss = [pid], set(), 0
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            rss += int(Path(f"/proc/{current}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            for task in Path(f"/proc/{current}/task").iterdir():
                todo.extend(map(int, (task / "children").read_text().split()))
        except (OSError, IndexError):
            pass
    return rss, len(seen)


def scratch_size(path):
    from .capacity import sizes
    return sum(sizes(Path(path)).values())


def execute(settings, scratch, source=None, module="markdown_for_llms.engines.marker", args=(), tick=None, label="process"):
    scratch = Path(scratch)
    write_json(scratch, label + "-limits.json", dict(memory_mib=settings.memory_mib, output_mib=settings.output_mib,
               timeout_seconds=settings.timeout_seconds, threads=settings.threads, process_limit=settings.process_limit))
    argv = command(settings, scratch, source, module, args, label)
    started = time.monotonic()
    selector = selectors.DefaultSelector()
    count, peak, peak_processes, reason = 0, 0, 0, None
    with (scratch / (label + ".log")).open("xb") as log:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env={}, close_fds=True, start_new_session=True)
        identity = process_identity(proc.pid)
        try:
            selector.register(proc.stdout, selectors.EVENT_READ)
            if tick and (tick_reason := tick(identity)):
                reason = tick_reason if isinstance(tick_reason, str) else "cancelled"
            while proc.poll() is None or selector.get_map():
                for key, _ in selector.select(0.1):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = settings.log_mib * 1048576 - count
                    log.write(chunk[:max(0, remaining)])
                    count += len(chunk)
                    if count > settings.log_mib * 1048576:
                        reason = "log_limit"
                rss, processes = tree_usage(proc.pid)
                peak, peak_processes = max(peak, rss), max(peak_processes, processes)
                if rss > settings.memory_mib * 1048576:
                    reason = "sampled_memory_limit"
                if processes > settings.process_limit:
                    reason = "process_limit"
                if time.monotonic() - started > settings.timeout_seconds:
                    reason = "timeout"
                if scratch_size(scratch) > settings.output_mib * 1048576:
                    reason = "output_limit"
                if tick and (tick_reason := tick(identity)):
                    reason = tick_reason if isinstance(tick_reason, str) else "cancelled"
                if reason and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=settings.grace_seconds)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            selector.close()
            proc.stdout.close()
    receipt = dict(exit=proc.returncode, reason=reason, elapsed_seconds=time.monotonic() - started,
                   peak_sampled_rss=peak, peak_sampled_processes=peak_processes,
                   log_bytes_observed=count, process=identity,
                   memory_mechanism="per-process RLIMIT_AS plus sampled aggregate RSS termination",
                   network="bubblewrap unshare-all; no host sockets", children_reaped=True)
    write_json(scratch, label + ".json", receipt)
    return receipt
