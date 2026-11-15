"""CLI of markdown-for-llms.

`mdf convert FILE --out DIR` converts one file directly with the conversion engine
(`markdown_for_llms.engine`), without the archive: `DIR` (new or empty) receives `document.md`,
`assets/` and `engine.json`, and stdout one JSON summary. Exit 0 for a complete conversion
(problems included, listed in the summary), 1 for a failure with its closed code, 2 for a usage
error. `mdf submit` sends files to the archive of the product (the same durable services as the
HTTP application) and `--wait` follows them. `mdf prepare --toolchain` builds the toolchain image
of the engine (the same function as `mdf bench prepare --toolchain`).
"""
import argparse
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import time

from .engine.codes import EngineError
from .engine.routes import CONVERTERS, OCR_MODES
from .services import Service, TERMINAL
from .settings import Settings
from .files import open_file


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


class ProgressLine:
    """Estimated progress of `submit --wait` on stderr, only on a terminal; stdout stays JSON."""
    def __init__(self, stream):
        self.stream, self.width = stream, 0
        self.enabled = stream.isatty()

    def show(self, job):
        if not self.enabled or not job.get('progress'):
            return
        filled = min(20, int(job['progress']['estimate_percent'] / 5))
        text = '[' + '#' * filled + '-' * (20 - filled) + '] ' + job['progress']['label']
        self.stream.write('\r' + text + ' ' * max(0, self.width - len(text)))
        self.stream.flush()
        self.width = len(text)

    def end(self):
        if self.enabled and self.width:
            self.stream.write('\n')
            self.stream.flush()
            self.width = 0


class StepLine:
    """The steps of `convert` on stderr, only on a terminal; stdout stays JSON."""
    def __init__(self, stream):
        self.stream, self.enabled = stream, stream.isatty()

    def __call__(self, event):
        if self.enabled and event.get("event") == "step":
            self.stream.write(event["step"] + "\n")
            self.stream.flush()
        return None


def summary(result, out):
    """The stdout of `mdf convert`: what the conversion did, without document text."""
    problems = result.get("problems") or []
    return {"status": result["status"], "error": (result.get("error") or {}).get("code"), "out": str(out),
            "source": {k: (result.get("source") or {}).get(k) for k in ("name", "format", "bytes", "sha256")},
            "route": {k: (result.get("route") or {}).get(k) for k in ("converter", "chosen_by", "reason")},
            "files": len(result.get("inventory") or []), "problem_counts": result.get("problem_counts"),
            "problem_codes": result.get("problem_codes"),
            "problems": problems[:20], "limits": result.get("limits") or [],
            "transit": result.get("transit"), "engine_json": str(Path(out) / "engine.json")}


def convert_file(args, settings):
    from .engine.core import convert
    limits = {"memory_mib": settings.memory_mib, "output_mib": settings.output_mib,
              "timeout_seconds": settings.timeout_seconds,
              "max_pages": settings.max_pages if args.max_pages is None else args.max_pages}
    if limits["max_pages"] < 1:
        emit({"error": "invalid --max-pages"})
        return 2
    request = {"route": args.route, "ocr": args.ocr, "lang": args.lang}
    try:
        result = convert(args.file, args.out, request, limits, work=args.work, observer=StepLine(sys.stderr))
    except EngineError as exc:          # refused before the conversion (the output or work directory)
        emit({"status": "refused", "error": exc.code.value, "out": args.out})
        return 2
    emit(summary(result, args.out))
    return 0 if result["status"] == "complete" else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["bench"]:
        # The bench has its own settings (MDF_BENCH_*): forward before the product settings load.
        from .bench.cli import main as bench_main
        return bench_main(argv[1:])
    if argv[:1] == ["prepare"]:
        # `mdf prepare --toolchain` is `mdf bench prepare --toolchain` (D8 of MFL032).
        from .bench.cli import main as bench_main
        return bench_main(argv)
    parser = argparse.ArgumentParser(prog="mdf")
    parser.add_argument("--env-file")
    parser.add_argument("--data-root")
    parser.add_argument("--model-root")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve")
    sub.add_parser("worker").add_argument("--once", action="store_true")
    sub.add_parser("jobs")
    admin = sub.add_parser('configure')
    admin.add_argument('--expected-config', required=True)
    admin.add_argument('--storage-mib', type=int)
    sub.add_parser('downgrade-storage').add_argument('--expected-config', required=True)
    convert = sub.add_parser("convert", help="convert one file with the engine into a directory, without the archive")
    convert.add_argument("file")
    convert.add_argument("--out", required=True, help="output directory, new or empty")
    convert.add_argument("--route", choices=CONVERTERS, help="a route of the matrix other than the default")
    convert.add_argument("--ocr", choices=OCR_MODES, default="auto")
    convert.add_argument("--lang", help="language of the document (recorded; no converter of this version uses it)")
    convert.add_argument("--max-pages", type=int, help="page limit (default: MDF_MAX_PAGES)")
    convert.add_argument("--work", help="directory for the private work of the conversion (default: a temporary one)")
    submit = sub.add_parser("submit", help="send files to the archive of the product")
    submit.add_argument("files", nargs="+")
    submit.add_argument("--wait", action="store_true")
    submit.add_argument("--force-ocr", action="store_true")
    for name in ("show", "retry", "cancel", "download"):
        p = sub.add_parser(name)
        p.add_argument("id")
        if name == "download":
            p.add_argument("--output", required=True)
            p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = Settings.load(args.env_file, {"data_root": args.data_root, "model_root": args.model_root})
        if args.command == "convert":
            return convert_file(args, settings)
        if args.command == "worker":
            from .worker import Worker
            worker = Worker(settings)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: setattr(worker, "stop", True))
            worker.run(args.once)
            return 0
        if args.command == "serve":
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
            return 0
        if args.command == 'configure':
            from .storage import Storage
            emit(Storage.configure(settings, args.expected_config, **{
                k: v for k, v in {'storage_mib': args.storage_mib}.items() if v is not None}))
            return 0
        if args.command == 'downgrade-storage':
            from .storage import Storage
            emit(Storage.downgrade(settings, args.expected_config))
            return 0
        service = Service(settings)
        if args.command == "jobs":
            emit(service.list())
        elif args.command in {"show", "retry", "cancel"}:
            emit(getattr(service, "get" if args.command == "show" else args.command)(args.id))
        elif args.command == "download":
            bundle = service.bundle(args.id)
            destination = Path(args.output).absolute()
            mode = "r+b" if args.overwrite and destination.exists() else "xb"
            with open_file(destination.parent, destination.name, mode) as output:
                if mode == "r+b":
                    output.truncate(0)
                with bundle.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
            emit({"path": args.output, "run": service.get(args.id)})
        elif args.command == "submit":
            failed = False
            for name in args.files:
                try:
                    with Path(name).open("rb") as stream:
                        job = service.submit(stream, Path(name).name, {"force_ocr": args.force_ocr})
                    if args.wait:
                        line = ProgressLine(sys.stderr)
                        try:
                            while job["state"] not in TERMINAL:
                                line.show(job)
                                time.sleep(settings.poll_seconds)
                                job = service.get(job["id"])
                        finally:
                            line.end()
                    emit(job)
                    failed |= job["state"] in {"failed", "interrupted", "cancelled"} or job["bundle"] in {"incomplete", "invalid"}
                except (ValueError, OSError, sqlite3.OperationalError) as exc:
                    failed = True
                    emit({"file": Path(name).name, "error": str(exc)})
            return int(failed)
        return 0
    except (ValueError, OSError, sqlite3.OperationalError) as exc:
        emit({"error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
