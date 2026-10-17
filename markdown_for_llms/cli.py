"""CLI using the same durable services as the HTTP application."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import time

from .services import Service, TERMINAL
from .settings import Settings
from .files import open_file


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None):
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
    admin.add_argument('--timeout-seconds', type=int)
    convert = sub.add_parser("convert")
    convert.add_argument("files", nargs="+")
    convert.add_argument("--wait", action="store_true")
    convert.add_argument("--force-ocr", action="store_true")
    for name in ("show", "retry", "cancel", "download"):
        p = sub.add_parser(name)
        p.add_argument("id")
        if name == "download":
            p.add_argument("--output", required=True)
            p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = Settings.load(args.env_file, {"data_root": args.data_root, "model_root": args.model_root})
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
                k: v for k, v in {'storage_mib': args.storage_mib,
                                  'timeout_seconds': args.timeout_seconds}.items() if v is not None}))
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
        elif args.command == "convert":
            failed = False
            for name in args.files:
                try:
                    with Path(name).open("rb") as stream:
                        job = service.submit(stream, Path(name).name, {"force_ocr": args.force_ocr})
                    if args.wait:
                        while job["state"] not in TERMINAL:
                            time.sleep(settings.poll_seconds)
                            job = service.get(job["id"])
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
