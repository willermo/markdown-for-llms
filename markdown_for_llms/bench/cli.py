"""Command line of `mdf bench` (forwarded by `markdown_for_llms.cli` before the product settings).

Exit codes: 0 the command did what was asked (for `run`: every attempt has an outcome, which may
be a failed attempt or `datalab_unresolved`); 2 some documents are unavailable; 1 failures,
caps or stops, a scan that found the key or could not read everything. Errors print a closed code (and, for the toolchain, a fixed hint), never document
text, keys or local paths. A fatal error (a tool container that outlived its timeout) stops the
command instead of one document.
"""
import argparse
import json
import sys

from .codes import BenchError, Code
from . import manifest as manifest_mod
from .settings import BenchSettings


def _common(parser):
    parser.add_argument("--env-file", help="file with MDF_BENCH_* settings; never export MDF_BENCH_* variables")
    parser.add_argument("--cache", help="cache directory (default: MDF_BENCH_CACHE_DIR or the XDG cache)")


def build_parser():
    parser = argparse.ArgumentParser(prog="mdf bench", description="Prepare the reference set (build the reference toolchain, fetch and derive "
                                     "the documents, generate and self-check the checks) and measure the routes of "
                                     "the conversion engine on it (`run`).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="prepare what the bench needs locally (also `mdf prepare`)")
    _common(p)
    p.add_argument("--toolchain", action="store_true",
                   help="build the reference toolchain image from the checkout (Docker), verify it and self-test it")
    p.add_argument("--context", default=None,
                   help="build context of the toolchain (default: scripts/bench-toolchain of the current directory)")
    p.add_argument("--no-cache", action="store_true", help="build without the builder cache")

    p = sub.add_parser("list", help="list the documents of the manifest")
    _common(p)
    p.add_argument("--level", choices=manifest_mod.LEVELS)
    p.add_argument("--manifest", help="another manifest file (default: the public reference set)")
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")

    p = sub.add_parser("fetch", help="download and verify the documents, their twins and derivations")
    _common(p)
    _selection(p)
    p.add_argument("--offline", action="store_true", help="verify the cache only, without network")

    p = sub.add_parser("checks", help="generate the checks from the twins (cache only) and, optionally, self-check them")
    _common(p)
    _selection(p)
    p.add_argument("--self-check", action="store_true", help="render each twin with a third-party tool and judge the checks")
    p.add_argument("--exclusions", help="exclusions file (development option; default: the packaged file)")
    p.add_argument("--json", dest="json_out", help="write the numbers (no document text) to this file")

    p = sub.add_parser("run", help="convert the documents on the routes of the engine and score the outputs")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--level", choices=manifest_mod.LEVELS)
    group.add_argument("--docs", help="comma-separated document ids, for example D04,D19")
    p.add_argument("--route", help="comma-separated routes (default: every local route of each format)")
    p.add_argument("--ocr", choices=("auto", "force"), help="OCR mode of the Marker route (default auto)")
    p.add_argument("--repeat", type=int, help="repetitions of each attempt (default 1, at most 10)")
    p.add_argument("--out", help="the new run directory, under the runs directory (default: <runs-dir>/<run id>)")
    p.add_argument("--runs-dir", help="runs directory (default: MDF_BENCH_RUNS_DIR or the XDG data directory)")
    p.add_argument("--manifest", help="another manifest file (default: the public reference set)")
    p.add_argument("--exclusions", help="exclusions file (development option; default: the packaged file)")
    p.add_argument("--cache", help="cache directory (default: MDF_BENCH_CACHE_DIR or the XDG cache)")
    p.add_argument("--resume", metavar="RUN_DIR", help="resume a run; every other option comes from its state, "
                   "except --confirm-transit, --env-file and --retry-unaccepted")
    p.add_argument("--datalab-mode", help="with --route datalab: comma-separated modes (lite, fast, balanced, accurate)")
    p.add_argument("--confirm-transit", action="store_true",
                   help="with --route datalab: confirm that the documents leave the machine for Datalab")
    p.add_argument("--max-cost-usd", help="with --route datalab: cap of the run, in USD (estimates until billed)")
    p.add_argument("--spend-cap-usd", help="with --route datalab: cap of the spend ledger of the runs folder, in USD "
                   "(default: MDF_BENCH_CAP_DATALAB_USD)")
    p.add_argument("--env-file", dest="key_file", help="with --route datalab: file with DATALAB_API_KEY (a regular "
                   "file of yours, mode 0600); it holds the key only, never MDF_BENCH_* settings")
    p.add_argument("--retry-unaccepted", action="store_true",
                   help="with --resume: convert again a Datalab attempt refused before its acceptance")

    p = sub.add_parser("report", help="write report.json and report.md of a finished run again from its saved "
                       "outputs, with the scoring of this version (nothing is converted or sent)")
    p.add_argument("run_dir", metavar="RUN_DIR")

    p = sub.add_parser("compare", help="compare two reports attempt by attempt")
    p.add_argument("report_a")
    p.add_argument("report_b")
    p.add_argument("--tolerance", type=float, default=0.005, help="fall of a ratio that is a regression (default 0.005)")
    p.add_argument("--fail-on-regression", action="store_true", help="exit with 1 when there is a regression")
    p.add_argument("--json", dest="json_out", help="write the comparison (numbers and codes) to this file")

    p = sub.add_parser("crosscheck", help="checks that every route fails in the same way (suspects)")
    p.add_argument("reports", nargs="+")
    p.add_argument("--out", help="directory of crosscheck.json and of the rendered pages")
    p.add_argument("--images", action="store_true", help="render the page of each suspect that names one")
    p.add_argument("--manifest", help="another manifest file (default: the public reference set)")
    p.add_argument("--cache", help="cache directory (default: MDF_BENCH_CACHE_DIR or the XDG cache)")

    p = sub.add_parser("secret-scan", help="search the Datalab key in files (it is never printed)")
    p.add_argument("--env-file", dest="key_file", required=True,
                   help="file with DATALAB_API_KEY (a regular file of yours, mode 0600)")
    p.add_argument("paths", nargs="+")
    return parser


def _selection(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--level", choices=manifest_mod.LEVELS)
    group.add_argument("--docs", help="comma-separated document ids, for example D04,D19")
    parser.add_argument("--manifest", help="another manifest file (default: the public reference set)")


def _selected(args):
    m = manifest_mod.load(args.manifest)
    ids = [x.strip() for x in args.docs.split(",") if x.strip()] if args.docs else None
    return m, m.select(level=args.level, ids=ids)


def cmd_fetch(args, settings):
    from .cache import Cache
    from .fetch import HttpTransport, resolve_document
    m, docs = _selected(args)
    cache = Cache(settings.cache_dir)
    transport = None if args.offline else HttpTransport()
    unavailable = 0
    for doc in docs:
        try:
            files = resolve_document(doc, m, cache, transport, offline=args.offline)
        except BenchError as exc:
            if exc.fatal:
                raise
            unavailable += 1
            print(f"{doc['id']} unavailable {exc.code.value}", flush=True)
            continue
        extra = " " + ",".join(w.value for w in files.warnings) if files.warnings else ""
        print(f"{doc['id']} available{extra}", flush=True)
    print(f"{len(docs) - unavailable} available, {unavailable} unavailable (manifest {m.sha256[:16]})")
    return 2 if unavailable else 0


def cmd_list(args, settings):
    m = manifest_mod.load(args.manifest)
    docs = m.select(level=args.level)
    if args.json:
        print(json.dumps({"manifest": {"sha256": m.sha256, "documents": len(m.documents)},
                          "documents": [{k: d[k] for k in ("id", "levels", "profile", "format", "pages", "title",
                                                           "language", "scripts")}
                                        | {"twin": _twin_kind(d)} for d in docs]},
                         ensure_ascii=False, indent=1))
        return 0
    print(f"manifest sha256 {m.sha256}, {len(docs)} of {len(m.documents)} documents")
    for d in docs:
        pages = d["pages"] if d["pages"] is not None else f"~{d['pages_estimate']}"
        print(f"{d['id']}  {','.join(d['levels']):<16} {d['profile']:<10} {d['format']:<5} {str(pages):>5}  "
              f"{_twin_kind(d):<15} {d['title']}")
    return 0


def _twin_kind(doc):
    """The kind of the twin, or `oracle` for a txt judged on its oracle alone."""
    return doc["twin"]["kind"] if doc["twin"] else "oracle"


def cmd_checks(args, settings):
    from .cache import Cache
    from .checks import load_or_generate
    from . import exclusions as excl
    from .fetch import resolve_document
    from .readers import read_twin
    from .selfcheck import run_self_check
    from ..engine import toolchain
    m, docs = _selected(args)
    cache = Cache(settings.cache_dir)
    data = excl.load(args.exclusions)
    out, unavailable, failed = {"manifest": m.sha256, "documents": []}, 0, 0
    for doc in docs:
        try:
            files = resolve_document(doc, m, cache, None, offline=True)
            if doc["twin"] is None:
                # judged on its oracle alone (`checks: "encoding"`): the derivation must give it back
                oracle = (files.hashes.get("derived") or {}).get("oracle_sha256")
                ok = oracle is not None and oracle == doc["input"]["reference"]["oracle_sha256"]
                failed += 0 if ok else 1
                print(f"{doc['id']} oracle {(oracle or '-')[:12]} {'OK' if ok else 'FAIL'}", flush=True)
                out["documents"].append({"id": doc["id"], "status": "oracle", "oracle_sha256": oracle, "verdict": ok})
                continue
            body = load_or_generate(doc, files, cache, lambda: read_twin(doc, files, cache))
        except BenchError as exc:
            if exc.fatal:
                raise
            unavailable += 1
            print(f"{doc['id']} unavailable {exc.code.value}", flush=True)
            out["documents"].append({"id": doc["id"], "status": "unavailable", "reason": exc.code.value})
            continue
        counts = {cls: sum(1 for c in body["checks"] if c["class"] == cls) for cls in ("content", "structure", "form")}
        record = {"id": doc["id"], "status": "checks", "checks_sha256": body["checks_sha256"], "counts": counts,
                  "not_determinable": body["not_determinable"], "stats": body["stats"]}
        line = (f"{doc['id']} checks {body['checks_sha256'][:12]} content {counts['content']} structure "
                f"{counts['structure']} form {counts['form']} nd {sum(body['not_determinable'].values())}")
        if args.self_check:
            try:
                sc = run_self_check(doc, files, cache, body, excl.for_document(data, doc["id"]))
            except BenchError as exc:
                if exc.fatal:
                    raise
                failed += 1
                print(line + f" self-check {exc.code.value}", flush=True)
                record["self_check"] = {"error": exc.code.value}
                out["documents"].append(record)
                continue
            record["self_check"] = sc
            failed += 0 if sc["verdict"] else 1
            ratios = " ".join(f"{cls[:4]} {_r(c['before'])}/{_r(c['after'])}" for cls, c in sc["classes"].items())
            low = [k for k, v in sc["coverage"].items() if v["ratio"] < 0.95]
            line += (f" | self-check {ratios} | coverage {'ok' if sc['coverage_ok'] else 'LOW ' + ','.join(low)}"
                     f" | omission {'ok' if sc['omission_ok'] else 'NOT DETECTED'}"
                     + (f" (not determinable: {','.join(sc['omission_not_determinable'])})" if sc["omission_not_determinable"] else "")
                     + f" | {'OK' if sc['verdict'] else 'FAIL'}")
        print(line, flush=True)
        out["documents"].append(record)
    out["toolchain"] = toolchain.report()
    if args.json_out:
        from pathlib import Path
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=1, sort_keys=True))
    return 1 if failed else 2 if unavailable else 0


RUN_OPTIONS = ("level", "docs", "route", "ocr", "repeat", "out", "runs_dir", "manifest", "exclusions", "cache",
               "datalab_mode", "max_cost_usd", "spend_cap_usd")


def cmd_run(args, settings):
    import signal
    from . import run
    # a run started in the background by a non-interactive shell inherits SIGINT ignored: the
    # default handler makes the signal a KeyboardInterrupt again, which stops the run cleanly
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if args.resume is not None:
        return run.resume(args.resume, confirm_transit=args.confirm_transit, key_file=args.key_file,
                          retry_unaccepted=args.retry_unaccepted)
    return run.start(settings, level=args.level, docs=args.docs, routes=args.route, ocr=args.ocr, repeat=args.repeat,
                     out=args.out, manifest=args.manifest, exclusions=args.exclusions,
                     datalab_modes=args.datalab_mode, confirm_transit=args.confirm_transit,
                     max_cost_usd=args.max_cost_usd, spend_cap_usd=args.spend_cap_usd, key_file=args.key_file,
                     retry_unaccepted=args.retry_unaccepted)


def cmd_report(args, settings):
    from . import run
    return run.rebuild_report(args.run_dir)


def cmd_compare(args, settings):
    from . import compare
    result = compare.compare(compare.load_report(args.report_a), compare.load_report(args.report_b), args.tolerance)
    compare.print_result(result)
    if args.json_out:
        from pathlib import Path
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=1, sort_keys=True))
    return 1 if args.fail_on_regression and result["regressions"] else 0


def cmd_crosscheck(args, settings):
    from . import crosscheck
    from .cache import Cache
    m = manifest_mod.load(args.manifest) if args.images else None
    result = crosscheck.run(args.reports, args.out, args.images, m, Cache(settings.cache_dir))
    for entry in result["documents"]:
        print(f"{entry['id']} paths {len(entry['paths'])} families {entry['families']} {entry['status']} "
              f"suspects {len(entry['suspects'])}", flush=True)
    by = ", ".join(f"{k} families {v}" for k, v in result["suspects_by_families"].items()) or "none"
    print(f"{len(result['documents'])} documents, {result['suspects']} suspects (by route families: {by})"
          + (f", {result['images']} pages rendered" if "images" in result else "")
          + "; variants of one route are correlated, not independent evidence")
    return 0


def cmd_secret_scan(args, settings):
    from . import secretscan
    read, errors, found = secretscan.scan(args.paths, secretscan.key_bytes(args.key_file))
    for path, n in found:
        print(f"found {n} in {path}")
    for path in errors:
        print(f"not read: {path}")
    print(f"secret-scan: {read} files read, {len(errors)} errors, {sum(n for _, n in found)} occurrences")
    if found:
        raise BenchError(Code.SECRET_FOUND, "the key is in a file")
    if errors:
        raise BenchError(Code.SECRET_SCAN_INCOMPLETE, "some paths could not be read")
    return 0


def _r(value):
    return "n/a" if value is None else f"{value:.3f}"


def cmd_prepare(args, settings):
    from pathlib import Path
    from .cache import Cache
    from .selftest import run_selftest
    from ..engine import toolchain
    if not args.toolchain:
        print("prepare: nothing to prepare (use --toolchain)", file=sys.stderr)
        return 2
    context = Path(args.context) if args.context else Path.cwd() / toolchain.DEFAULT_CONTEXT
    toolchain.prepare(context, Cache(settings.cache_dir), no_cache=args.no_cache, selftest=run_selftest)
    return 0


COMMANDS = {"prepare": cmd_prepare, "list": cmd_list, "fetch": cmd_fetch, "checks": cmd_checks, "run": cmd_run,
            "report": cmd_report, "compare": cmd_compare, "crosscheck": cmd_crosscheck, "secret-scan": cmd_secret_scan}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run" and args.resume is not None and any(getattr(args, k) is not None for k in RUN_OPTIONS):
            raise BenchError(Code.RESUME_OPTION_NOT_ALLOWED, "--resume takes its options from the state of the run")
        # `run` and `secret-scan` read no settings file: their settings come from the environment and the
        # options; their `--env-file` (`key_file`) is the file of the Datalab key alone, never opened here (Δ5)
        settings = BenchSettings.load(getattr(args, "env_file", None), cache=getattr(args, "cache", None),
                                      runs=getattr(args, "runs_dir", None))
        from ..engine import tools
        tools.set_memory_limit(settings.tool_memory_mb)
        return COMMANDS[args.command](args, settings)
    except BenchError as exc:
        from ..engine.toolchain import HINTS
        hint = HINTS.get(exc.code)
        print(f"error: {exc.code.value}" + (f" ({hint})" if hint else ""), file=sys.stderr)
        return 1
