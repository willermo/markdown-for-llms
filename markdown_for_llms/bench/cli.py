"""Command line of `mdf bench` (forwarded by `markdown_for_llms.cli` before the product settings).

Exit codes: 0 everything requested was done; 2 some documents are unavailable; 1 failures,
caps or stops. Errors print a closed code (and, for the toolchain, a fixed hint), never document
text, keys or local paths. A fatal error (a tool container that outlived its timeout) stops the
command instead of one document.
"""
import argparse
import json
import sys

from .codes import BenchError
from . import manifest as manifest_mod
from .settings import BenchSettings


def _common(parser):
    parser.add_argument("--env-file", help="file with MDF_BENCH_* settings; never export MDF_BENCH_* variables")
    parser.add_argument("--cache", help="cache directory (default: MDF_BENCH_CACHE_DIR or the XDG cache)")


def build_parser():
    parser = argparse.ArgumentParser(prog="mdf bench", description="Prepare the reference set: build the reference toolchain, fetch and derive the "
                                     "documents, generate and self-check the checks. It converts nothing yet.")
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
                          "documents": [{k: d[k] for k in ("id", "levels", "profile", "format", "pages", "title")}
                                        | {"twin": d["twin"]["kind"]} for d in docs]},
                         ensure_ascii=False, indent=1))
        return 0
    print(f"manifest sha256 {m.sha256}, {len(docs)} of {len(m.documents)} documents")
    for d in docs:
        pages = d["pages"] if d["pages"] is not None else f"~{d['pages_estimate']}"
        print(f"{d['id']}  {','.join(d['levels']):<16} {d['profile']:<10} {d['format']:<5} {str(pages):>5}  "
              f"{d['twin']['kind']:<15} {d['title']}")
    return 0


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


COMMANDS = {"prepare": cmd_prepare, "list": cmd_list, "fetch": cmd_fetch, "checks": cmd_checks}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        settings = BenchSettings.load(getattr(args, "env_file", None), cache=getattr(args, "cache", None))
        from ..engine import tools
        tools.set_memory_limit(settings.tool_memory_mb)
        return COMMANDS[args.command](args, settings)
    except BenchError as exc:
        from ..engine.toolchain import HINTS
        hint = HINTS.get(exc.code)
        print(f"error: {exc.code.value}" + (f" ({hint})" if hint else ""), file=sys.stderr)
        return 1
