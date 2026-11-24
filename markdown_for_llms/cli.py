"""CLI of markdown-for-llms.

`mdf convert FILE --out DIR` converts one file directly with the conversion engine
(`markdown_for_llms.engine`), without the archive: `DIR` (new or empty) receives `document.md`,
`assets/` and `engine.json`, and stdout one JSON summary. Exit 0 for a complete conversion
(problems included, listed in the summary), 1 for a failure with its closed code, 2 for a usage
error. With several files `DIR` is a new or empty directory with one numbered directory per file,
and stdout a JSON list (exit 1 when one of them is not complete). `--route marker-server
--server-mode {balanced,accurate}` converts with Marker and the inference of Surya on the server
named by `--server` in `servers.json` (`--servers FILE`; MFL033): `--confirm-transit` for a
server outside the loopback; for an endpoint with a provider (RunPod, Modal) also
`--server-cap-usd`, `--spend-ledger FILE` and `--spend-cap-usd`, inside a live session of
`scripts/servers/session.py run` (`server_ledger_required`, `server_not_armed`,
`session_orphaned`). Its keys come from `--env-file` or the environment. `mdf submit` sends files to the archive of the product (the same durable services as the
HTTP application) and `--wait` follows them. `mdf prepare --toolchain` builds the toolchain image
of the engine (the same function as `mdf bench prepare --toolchain`); `mdf prepare --weights`
downloads and checks the weights of the Marker route (`engine/weights.py`); `mdf prepare
--cpu-server` downloads and checks the local CPU server of the marker-server route
(`engine/cpuserver.py`).
"""
import argparse
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import sqlite3
import sys
import time

from .engine.codes import EngineError
from .engine.routes import CONVERTERS, DATALAB_MODES, MARKER_SERVER, OCR_MODES, SERVER_MODES
from .services import Refused as ServiceRefused, Service, TERMINAL
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
            "transit": result.get("transit"), "cost": result.get("cost"),
            "engine_json": str(Path(out) / "engine.json")}


SERVER_OPTIONS = ("server_mode", "server", "servers", "server_cap_usd", "spend_ledger", "spend_cap_usd")
SESSION_HINT = ("scripts/servers/session.py run --provider PROVIDER --name ENDPOINT --cap-usd X --spend-ledger FILE "
                "--spend-cap-usd Y --env-file .env -- mdf convert ...")


class Refused(Exception):
    def __init__(self, code, status=1, hint=None):
        super().__init__(code)
        self.code, self.status, self.hint = code, status, hint


def _usd(value, what):
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        amount = None
    if amount is None or not amount.is_finite() or not 0 <= amount <= 1000:
        raise Refused("invalid_setting", 2, what + ": an amount in USD from 0 to 1000")
    return amount


def prepare_server(args):
    """The endpoint, its keys and the server options of the request; for a provider the session,
    the budget and the reservation hook (MFL033). Refused with a closed code before anything is
    converted."""
    from .engine import servers
    from .gpu import presets
    ep = servers.load(args.servers, args.server, presets=presets.endpoint_table())
    keys = servers.read_keys(ep, args.env_file, os.environ)
    srv = {"name": ep.name, "mode": args.server_mode, "transit_confirmed": bool(args.confirm_transit)}
    if not ep.costs:
        if any(getattr(args, k) is not None for k in ("server_cap_usd", "spend_ledger", "spend_cap_usd")):
            raise Refused("route_option_unsupported", 2, "the spending options are for an endpoint with a provider")
        return ep, keys, srv, None
    if args.server_cap_usd is None or args.spend_ledger is None or args.spend_cap_usd is None:
        raise Refused("server_ledger_required", 2, "--server-cap-usd, --spend-ledger and --spend-cap-usd")
    cap = _usd(args.server_cap_usd, "--server-cap-usd") * 100
    spend_cap = _usd(args.spend_cap_usd, "--spend-cap-usd") * 100
    from .bench import serverspend, spend
    from .bench.codes import BenchError
    ledger = spend.Ledger.file(args.spend_ledger)
    try:
        session = serverspend.session(ledger.path, ep.provider, servers.canonical_url(ep.url), os.environ,
                                      rate=ep.rate_usd_per_s)
        total = Decimal(serverspend.summary(spend.read_only(ledger.path), ep.provider, ledger.path)["total_cents"])
    except BenchError as exc:
        hint = SESSION_HINT if exc.code.value in ("server_not_armed", "session_orphaned") else None
        if exc.code.value == "server_not_configured":
            hint = exc.detail                         # the rate of servers.json and of the arm (S-B6): no key
        raise Refused(exc.code.value, 1, hint) from None
    if total > spend_cap:
        raise Refused("server_cap", 1, "the ledger of %s is over --spend-cap-usd" % ep.provider)
    budget = min(cap, Decimal(session["window_left_cents"]))
    srv.update(budget_cents=str(budget), session={"arm_id": session["arm_id"], "armed_until": session["armed_until"]})
    return ep, keys, srv, (ledger, ep.provider, session["arm_id"], cap)


class Chain:
    def __init__(self, *observers):
        self.observers = [o for o in observers if o is not None]

    def __call__(self, event):
        for observer in self.observers:
            stop = observer(event)
            if stop:
                return stop
        return None


def convert_file(args, settings, key=None, server=None):
    from .engine.core import convert
    limits = settings.engine_limits(args.max_pages)
    if limits["max_pages"] < 1:
        emit({"error": "invalid --max-pages"})
        return 2
    request = {"route": args.route, "ocr": args.ocr, "lang": args.lang}
    if args.route == "datalab":
        # the document leaves the machine: only with --confirm-transit, under --datalab-cap-usd;
        # without the confirmation the engine stops before sending, with the pages and the estimate
        request["datalab"] = {"mode": args.datalab_mode, "transit_confirmed": args.confirm_transit,
                              "cap_usd": args.datalab_cap_usd}
    elif args.route == MARKER_SERVER:
        if server is None:
            emit({"status": "refused", "error": "server_not_configured", "out": args.out})
            return 1
    elif args.confirm_transit or args.datalab_mode or args.datalab_cap_usd is not None \
            or any(getattr(args, k) is not None for k in SERVER_OPTIONS):
        emit({"status": "refused", "error": "route_option_unsupported", "out": args.out})
        return 2
    files = args.file
    if len(files) == 1:
        targets = [(files[0], Path(args.out))]
    else:
        out = Path(args.out)
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            emit({"status": "refused", "error": "output_invalid", "out": args.out, "detail": "a new or empty directory"})
            return 2
        targets = [(f, out / ("%03d-%s" % (i, Path(f).stem[:40] or "file"))) for i, f in enumerate(files, 1)]
    summaries, failed = [], False
    for source, out in targets:
        attempt = secrets.token_hex(8)
        hook = None
        extra = {}
        if server is not None:
            ep, keys, srv, rented = server
            extra = {"server_endpoint": ep, "server_keys": keys}
            request = dict(request, server=dict(srv))
            if rented is not None:
                from .bench.serverspend import Reservation
                ledger, provider, arm_id, cap = rented
                hook = Reservation(ledger, provider, arm_id, "mdf-convert", attempt, cap)
        try:
            result = convert(source, out, dict(request, attempt=attempt), limits, work=args.work,
                             observer=Chain(hook, StepLine(sys.stderr)), model_root=settings.model_root,
                             datalab_key=key, **extra)
        except EngineError as exc:      # refused before the conversion (the output or work directory)
            emit({"status": "refused", "error": exc.code.value, "out": str(out)})
            return 2
        if hook is not None:
            hook.complete(result)
        summaries.append(summary(result, out))
        failed = failed or result["status"] != "complete"
    emit(summaries[0] if len(summaries) == 1 else summaries)
    return 1 if failed else 0


def prepare_weights(argv):
    """`mdf prepare --weights [--model-root DIR]`: the only download of the Marker weights."""
    from .engine import weights
    parser = argparse.ArgumentParser(prog="mdf prepare")
    parser.add_argument("--weights", action="store_true", required=True)
    parser.add_argument("--model-root", help="model root (default: %s)" % weights.default_root())
    args = parser.parse_args(argv)
    try:
        emit(weights.prepare(args.model_root or weights.default_root()))
    except EngineError as exc:
        emit({"status": "failed", "error": exc.code.value, "detail": exc.detail})
        return 1
    return 0


def prepare_cpu_server(argv):
    """`mdf prepare --cpu-server [--root DIR]`: the only download of the local CPU server of the
    marker-server route (llama-server and the GGUF weights of Surya, `engine/cpuserver.py`). Exit 1
    also when the build cannot bind a Unix socket (`unix_socket: false`)."""
    from .engine import cpuserver
    parser = argparse.ArgumentParser(prog="mdf prepare")
    parser.add_argument("--cpu-server", action="store_true", required=True)
    parser.add_argument("--root", help="directory of the CPU server (default: %s)" % cpuserver.default_root())
    args = parser.parse_args(argv)
    try:
        result = cpuserver.prepare(args.root)
    except EngineError as exc:
        emit({"status": "failed", "error": exc.code.value, "detail": exc.detail})
        return 1
    emit(result)
    return 0 if result.get("unix_socket") else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["bench"]:
        # The bench has its own settings (MDF_BENCH_*): forward before the product settings load.
        from .bench.cli import main as bench_main
        return bench_main(argv[1:])
    if argv[:1] == ["prepare"] and "--weights" in argv:
        return prepare_weights(argv[1:])
    if argv[:1] == ["prepare"] and "--cpu-server" in argv:
        return prepare_cpu_server(argv[1:])
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
    downgrade = sub.add_parser('downgrade-storage')
    downgrade.add_argument('--expected-config', required=True)
    downgrade.add_argument('--to-schema', type=int, default=2,
                           help='database schema to return to, from 2 (the default) to the one before the current '
                                'schema; 3 for the binary of schema 3')
    convert = sub.add_parser("convert", help="convert one file with the engine into a directory, without the archive")
    convert.add_argument("file", nargs="+")
    convert.add_argument("--out", required=True, help="output directory, new or empty (with several files: one "
                                                      "numbered directory per file inside it)")
    convert.add_argument("--route", choices=CONVERTERS, help="a route of the matrix other than the default")
    convert.add_argument("--ocr", choices=OCR_MODES, default="auto")
    convert.add_argument("--lang", help="language of the document (recorded; no converter of this version uses it)")
    convert.add_argument("--max-pages", type=int, help="page limit (default: MDF_MAX_PAGES)")
    convert.add_argument("--work", help="directory for the private work of the conversion (default: a temporary one)")
    convert.add_argument("--datalab-mode", choices=DATALAB_MODES,
                         help="mode of the Datalab route (default: balanced)")
    convert.add_argument("--datalab-cap-usd", default=None,
                         help="spending cap of this conversion with Datalab, in USD (default: 0, nothing is sent)")
    convert.add_argument("--confirm-transit", action="store_true",
                         help="confirm that the document leaves the machine (Datalab), or that its page images go to "
                              "an inference server outside the loopback (marker-server)")
    convert.add_argument("--server-mode", choices=SERVER_MODES, help="with --route marker-server: balanced or accurate")
    convert.add_argument("--server", help="with --route marker-server: an endpoint of servers.json (default: its default)")
    convert.add_argument("--servers", help="with --route marker-server: the servers.json file (default: "
                                           "$XDG_CONFIG_HOME/markdown-for-llms/servers.json)")
    convert.add_argument("--server-cap-usd", help="with an endpoint with a provider: cap of this conversion, in USD")
    convert.add_argument("--spend-ledger", help="with an endpoint with a provider: the spend ledger (spend.jsonl)")
    convert.add_argument("--spend-cap-usd", help="with an endpoint with a provider: cap of the ledger for the provider, in USD")
    submit = sub.add_parser("submit", help="send files to the archive of the product")
    submit.add_argument("files", nargs="+")
    submit.add_argument("--wait", action="store_true")
    submit.add_argument("--route", choices=CONVERTERS, help="a route of the matrix other than the default")
    submit.add_argument("--ocr", choices=OCR_MODES, help="OCR of the job (default: auto)")
    submit.add_argument("--lang", help="language of the document (recorded; no converter of this version uses it)")
    submit.add_argument("--datalab-mode", choices=DATALAB_MODES, help="mode of the Datalab route (default: balanced)")
    submit.add_argument("--confirm-transit", action="store_true",
                        help="confirm that the document leaves the machine: for Datalab (the worker applies "
                             "MDF_DATALAB_CAP_USD), or its page images for the rented GPU of a GPU preset (the worker "
                             "applies the GPU caps)")
    submit.add_argument("--preset", help="a conversion preset (mdf presets list): its route and options, with only "
                                         "--lang and --confirm-transit beside it")
    presets = sub.add_parser("presets", help="the conversion presets of the instance and your own")
    psub = presets.add_subparsers(dest="presets_command", required=True)
    psub.add_parser("list")
    psub.add_parser("show").add_argument("name")
    add = psub.add_parser("add", help="add a conversion preset of your own")
    add.add_argument("name")
    add.add_argument("--route", choices=CONVERTERS)
    add.add_argument("--server-mode", choices=SERVER_MODES)
    add.add_argument("--gpu-preset")
    add.add_argument("--ocr", choices=OCR_MODES)
    add.add_argument("--datalab-mode", choices=DATALAB_MODES)
    psub.add_parser("remove").add_argument("name")
    default = psub.add_parser("default", help="set your default conversion preset, or clear it")
    choice = default.add_mutually_exclusive_group(required=True)
    choice.add_argument("name", nargs="?")
    choice.add_argument("--clear", action="store_true")
    gpu = sub.add_parser("gpu", help="the GPU session of the archive: its state, an estimate, arm ahead, disarm")
    gsub = gpu.add_subparsers(dest="gpu_command", required=True)
    gsub.add_parser("session")
    gsub.add_parser("estimate").add_argument("preset")
    arm = gsub.add_parser("arm", help="arm a GPU preset ahead (RunPod): the worker opens its session between two jobs")
    arm.add_argument("preset")
    arm.add_argument("--confirm-transit", action="store_true", help="confirm that the page images leave the machine")
    gsub.add_parser("disarm", help="stop the GPU session of a preset; its jobs in progress and queued are cancelled"
                    ).add_argument("preset")
    for name in ("show", "retry", "cancel", "download"):
        p = sub.add_parser(name)
        p.add_argument("id")
        if name == "retry":
            # a retry of a Datalab or GPU run sends the document (or its page images) again: the transit is confirmed again
            p.add_argument("--confirm-transit", action="store_true",
                           help="confirm again that the document leaves the machine: for Datalab, or its page images "
                                "for the rented GPU of a GPU preset")
        if name == "download":
            p.add_argument("--output", required=True)
            p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    key = server = None
    if args.command == "convert" and args.route == MARKER_SERVER:
        # as the Datalab key: the endpoint, its keys and the session before anything else reads the env file
        if args.datalab_mode or args.datalab_cap_usd is not None or args.server_mode is None:
            emit({"status": "refused", "error": "route_option_unsupported", "out": args.out,
                  "detail": "--server-mode balanced or accurate, without Datalab options"})
            return 2
        try:
            server = prepare_server(args)
        except Refused as exc:
            emit({"status": "refused", "error": exc.code, "out": args.out, **({"detail": exc.hint} if exc.hint else {})})
            return exc.status
        except EngineError as exc:
            emit({"status": "refused", "error": exc.code.value, "out": args.out})
            return 1
    if args.command == "convert" and args.route == "datalab":
        # the key file is checked (regular, of the caller, 0600, no link) before anything else
        # reads it, Settings.load included (HC2 of the review of B of MFL032)
        from .engine import datalab
        try:
            key = datalab.read_key(args.env_file)
        except EngineError as exc:
            emit({"status": "refused", "error": exc.code.value, "out": args.out})
            return 1
    if args.command == "worker" and args.env_file is not None:
        # the env file of the worker holds the keys of its GPU sessions (MFL023): checked as envfile.read
        # checks a key file (regular, of the caller, 0600, no link, no blocking) before anything reads it
        from .engine import envfile
        try:
            envfile.read(args.env_file, [])
        except envfile.Unsafe as exc:
            emit({"error": str(exc), "code": "server_key_unsafe"})
            return 1
    try:
        settings = Settings.load(args.env_file, {"data_root": args.data_root, "model_root": args.model_root})
        if args.command == "convert":
            # SIGTERM as SIGINT: the `finally` of the engine stops the children, their scopes and the
            # local CPU server, and removes the private socket directory (MFL033, plan r2 §7)
            signal.signal(signal.SIGTERM, signal.default_int_handler)
            try:
                return convert_file(args, settings, key, server)
            except KeyboardInterrupt:
                emit({"status": "interrupted", "error": "interrupted", "out": args.out})
                return 130
        if args.command == "worker":
            from .worker import Worker
            worker = Worker(settings, env_file=args.env_file)
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
            emit(Storage.downgrade(settings, args.expected_config, args.to_schema))
            return 0
        service = Service(settings)
        if args.command == "jobs":
            emit(service.list())
        elif args.command == "presets":
            name = getattr(args, "name", None)
            if args.presets_command == "list":
                emit(service.presets())
            elif args.presets_command == "show":
                emit(service.preset(name))
            elif args.presets_command == "add":
                emit(service.add_preset(name, {k: getattr(args, k) for k in
                                               ("route", "server_mode", "gpu_preset", "ocr", "datalab_mode")}))
            elif args.presets_command == "remove":
                emit(service.remove_preset(name))
            else:
                emit(service.set_default(None if args.clear else name))
        elif args.command == "gpu":
            # the same services as /api/v1/gpu/* (MFL023 B5)
            if args.gpu_command == "session":
                emit(service.gpu_session())
            elif args.gpu_command == "estimate":
                emit(service.gpu_estimate(args.preset))
            elif args.gpu_command == "arm":
                emit(service.gpu_arm(args.preset, bool(args.confirm_transit)))
            else:
                emit(service.gpu_disarm(args.preset))
        elif args.command == "retry":
            # the same as POST /api/v1/runs/ID/retry?transit_confirmed=true
            emit(service.retry(args.id, {"transit_confirmed": True} if args.confirm_transit else None))
        elif args.command in {"show", "cancel"}:
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
                    if args.preset is not None:
                        if args.route or args.ocr or args.datalab_mode:
                            raise ServiceRefused("invalid_setting", "--preset takes only --lang and --confirm-transit")
                        options = service.preset_options(args.preset, dict(
                            {"lang": args.lang} if args.lang else {}, transit_confirmed=args.confirm_transit))
                    else:
                        options = {"route": args.route, "ocr": args.ocr or "auto", "lang": args.lang,
                                   "datalab_mode": args.datalab_mode, "transit_confirmed": args.confirm_transit}
                    with Path(name).open("rb") as stream:
                        job = service.submit(stream, Path(name).name, options)
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
                    emit({"file": Path(name).name, "error": str(exc), **({"code": exc.code} if getattr(exc, "code", None) else {})})
            return int(failed)
        return 0
    except (ValueError, OSError, sqlite3.OperationalError) as exc:
        emit({"error": str(exc), **({"code": exc.code} if getattr(exc, "code", None) else {})})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
