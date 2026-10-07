#!/usr/bin/env python3
"""V0: script originali e contenuti integrali, eseguibile solo dopo R/S.

Non installa pacchetti e non modifica il prodotto. La copia dei sorgenti è una
baseline identificata, mai una prova della futura wheel. I figli script diretti
usano il path fidato della copia come il legacy; nessun PYTHONPATH aggiunto.
"""

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace


LEGACY_FILES = ("tests/unit/test_cleaning.py", "tests/unit/test_validation.py",
                "tests/unit/test_chunking.py", "tests/integration/test_pipeline.py")
LEGACY_FAILURES = tuple("tests/integration/test_pipeline.py::TestPipelineIntegration::" + name for name in (
    "test_end_to_end_without_conversion", "test_pipeline_statistics_tracking",
    "test_concurrent_processing", "test_pipeline_logging", "test_configuration_integration"))
LEGACY_NODEIDS = "temp/run-a001-fase0-uv/evidence/implementation-r001/resume-baseline-s003/legacy-post-nodeids.json"


def f4_references(source, text):
    """Scanner circoscritto alla fixture F4: nessuna riscrittura del Markdown."""
    first, second = "Primo esperimento", "Secondo esperimento"
    source_contexts = ("Il riferimento [1] segue il primo risultato.",
                       '<p id="riferimento">[1] Autore sintetico, Titolo, 2026.</p>')
    if any(source.count(value) != 1 for value in source_contexts) or source.count("[1]") != 2:
        raise ValueError("F4: riferimenti sorgente inattesi")
    # Alternativa esatta, non unescape globale; esclusi marker parzialmente escaped.
    marker = r"(?:\\\[1\\\]|(?<![\\\[])\[1\](?!\]))"
    patterns = (r"Il riferimento (" + marker + r") segue il primo risultato\.",
                r"(?m)^(" + marker + r") Autore sintetico, Titolo, 2026\.$")
    matches = [list(re.finditer(pattern, text)) for pattern in patterns]
    if any(len(items) != 1 for items in matches):
        raise ValueError("F4: richiamo/bibliografia mancanti, diversi o duplicati")
    spans = [items[0].span(1) for items in matches]
    all_markers = [m.span() for m in re.finditer(r"(?:\\\[\d+\\\]|(?<![\\\[])\[\d+\](?!\]))", text)]
    if all_markers != spans:
        raise ValueError("F4: quantità/ordine/contesti riferimenti difformi")
    for document, positions in ((source, [source.index(c) for c in source_contexts]),
                                (text, [s[0] for s in spans])):
        if (document.count(first) != 1 or document.count(second) != 1 or
                not document.index(first) < positions[0] < document.index(second) < positions[1]):
            raise ValueError("F4: sezioni assenti o riferimenti fuori contesto/ordine")
    return {"source_count": 2, "output_count": 2, "identity": "1", "contexts_order_match": True,
            "source_char_positions": [source.index("[1]", source.index(c)) for c in source_contexts],
            "output_char_spans": spans, "output_markers": [text[a:b] for a, b in spans]}


def sliding_audit(content, tokens, token_bytes, windows, chunks, size=1000, overlap=100):
    """Controllo puro: ID osservati, slice decoded e testo, limitato alla F6 ASCII."""
    if not content.isascii() or not 0 < overlap < size or not tokens:
        raise ValueError("Sliding: richiesti sorgente ASCII e parametri con overlap positivo")
    if len(tokens) != len(token_bytes) or any(type(t) is not int for t in tokens):
        raise ValueError("Sliding: ID/byte dei token incompleti")
    offsets = [0]
    vocabulary = {}
    for token, data in zip(tokens, token_bytes):
        if not data or not data.isascii() or vocabulary.setdefault(token, data) != data:
            raise ValueError("Sliding: byte/identità token incoerenti")
        offsets.append(offsets[-1] + len(data))
    if b"".join(token_bytes) != content.encode("ascii"):
        raise ValueError("Sliding: token non ricostruiscono il sorgente integrale")
    expected = []
    for start in range(0, len(tokens), size - overlap):
        end = min(start + size, len(tokens))
        expected.append((start, end))
        if end == len(tokens):
            break
    if len(chunks) < 2 or len(chunks) != len(expected) or len(windows) != len(expected):
        raise ValueError("Sliding: quantità finestre/chunk difforme")
    bodies = []
    for (start, end), window, chunk in zip(expected, windows, chunks):
        if (window["start_token"], window["end_token"]) != (start, end) or window["source_tokens"] != tokens[start:end]:
            raise ValueError("Sliding: ordine/indici/ID finestra difformi")
        raw = b"".join(token_bytes[start:end]).decode("ascii")
        if window.get("decoded", raw) != raw or chunk != raw.strip() or not chunk:
            raise ValueError("Sliding: corpo/decode della slice difforme")
        left = len(raw) - len(raw.lstrip())
        right = len(raw) - len(raw.rstrip())
        a, b = offsets[start] + left, offsets[end] - right
        if content[a:b] != chunk:
            raise ValueError("Sliding: posizione del corpo nel sorgente difforme")
        bodies.append({"start_char": a, "end_char": b, "start_byte": a, "end_byte": b,
                       "raw_start_byte": offsets[start], "raw_end_byte": offsets[end],
                       "excluded_prefix": raw[:left], "excluded_suffix": raw[len(raw)-right:] if right else ""})
    intersections = []
    for i, (left, right) in enumerate(zip(bodies, bodies[1:])):
        start, end = expected[i+1][0], expected[i][1]
        ids = tokens[start:end]
        if not ids or windows[i]["source_tokens"][-len(ids):] != ids or windows[i+1]["source_tokens"][:len(ids)] != ids:
            raise ValueError("Sliding: intersezione ID assente o alterata")
        a, b = right["start_char"], left["end_char"]
        if not left["start_char"] < a < b < right["end_char"]:
            raise ValueError("Sliding: gap/intersezione/ordine del testo difforme")
        shared = content[a:b]
        if chunks[i][-len(shared):] != shared or chunks[i+1][:len(shared)] != shared:
            raise ValueError("Sliding: suffix/prefix effettivo difforme dal sorgente")
        if not offsets[start] <= a < b <= offsets[end]:
            raise ValueError("Sliding: testo fuori dall'intersezione token")
        intersections.append({"start_token": start, "end_token": end, "source_ids": ids,
                              "token_count": len(ids), "start_byte": a, "end_byte": b,
                              "start_char": a, "end_char": b, "shared_text": shared,
                              "characters": len(shared), "source_ids_match": True,
                              "actual_suffix_prefix_match": True})
    leading, trailing = content[:bodies[0]["start_char"]], content[bodies[-1]["end_char"]:]
    if leading.strip() or trailing.strip() or bodies[-1]["raw_end_byte"] != len(content):
        raise ValueError("Sliding: copertura/fine sorgente incompleta")
    return {"bodies": bodies, "intersections": intersections, "continuous_coverage": True,
            "source_end_byte": len(content), "excluded_leading": leading, "excluded_trailing": trailing,
            "multi_chunk": True, "nonempty_overlap": True, "scope": "F6 ASCII; byte offsets = char offsets"}


def legacy_pytest_args(repo, cache, phase):
    if phase not in ("collection", "run") or not cache.is_absolute():
        raise ValueError("Suite: fase/cache invalida")
    display = (["--collect-only", "-q", "--verbosity=-1"] if phase == "collection" else
               ["-v", "-rA", "--verbosity=1", "-o", "verbosity_test_cases=1"])
    config = ['-c',str(repo/'baseline-pytest.ini')] if (repo/'baseline-pytest.ini').exists() else []
    return [str(repo / name) for name in LEGACY_FILES] + display + config + ["-o", "cache_dir=" + str(cache)]


def cache_inventory(path):
    if path.is_symlink():
        raise ValueError("Suite: cache symlink")
    records = []
    for p in sorted(path.rglob("*")) if path.exists() else []:
        if p.is_symlink():
            raise ValueError("Suite: elemento cache symlink")
        if p.is_file():
            records.append(file_record(p, path))
    values = {}
    for name in ("nodeids", "lastfailed"):
        p = path / "v/cache" / name
        values[name] = json.loads(p.read_text()) if p.exists() else None
    return {"path": str(path), "exists": path.exists(), "files": records, "values": values}


def unique_nodeids(items):
    if not isinstance(items, list) or any(not isinstance(x, str) or x.split("::")[0] not in LEGACY_FILES or "::" not in x for x in items):
        raise ValueError("Suite: nodeid fuori selezione")
    if len(set(items)) != len(items):
        raise ValueError("Suite: nodeid duplicato")
    return set(items)


def compare_legacy_config(collection, execution, repo, cwd):
    """Confronto puro della baseline senza config file, con override esatti per fase.

    Repo e CWD sono parametri fidati dell'invocazione, non valori ricavati dagli
    observed. I raw restano intatti; soltanto dopo la validazione si separano
    gli override dal comune. Nessun accesso al filesystem o import di pytest.
    """
    for path in (repo, cwd):
        if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
            raise ValueError("Suite: repo/CWD attesi devono essere assoluti e canonici")
    if cwd.name != "legacy-suite":
        raise ValueError("Suite: CWD attesa fuori dal workspace legacy-suite")

    def required(mapping, key, typ):
        if not isinstance(mapping, dict) or key not in mapping or type(mapping[key]) is not typ:
            raise ValueError("Suite: campo obbligatorio assente/tipo errato: " + key)
        return mapping[key]

    def strings(mapping, key):
        value = required(mapping, key, list)
        if any(type(item) is not str for item in value):
            raise ValueError("Suite: lista di stringhe richiesta: " + key)
        return value

    common = []
    for phase, data in (("collection", collection), ("run", execution)):
        if required(data, "phase", str) != phase or required(data, "cwd", str) != str(cwd):
            raise ValueError("Suite: fase/CWD discordanti dall'invocazione")
        config = required(data, "config", dict)
        cache = cwd / ("pytest-cache-" + phase)
        argv = legacy_pytest_args(repo, cache, phase)
        if strings(data, "argv") != argv or strings(config, "invocation_args") != argv:
            raise ValueError("Suite: argv/override discordanti dalla fase prevista")
        if required(config, "rootdir", str) != str(repo):
            raise ValueError("Suite: rootdir diverso dal repo atteso")
        # Null esplicito è l'assenza osservata di un config file, non un campo
        # dimenticato dal producer. Un file/record/hash nuovo invalida baseline.
        empty_config = repo/'baseline-pytest.ini'
        if empty_config.exists():
            if empty_config.read_bytes() != b'':
                raise ValueError('Suite: configurazione baseline deve essere vuota')
            if (required(config,'configfile',str) != str(empty_config) or
                    required(config,'configfile_record',dict) != file_record(empty_config,repo)):
                raise ValueError('Suite: config baseline diversa dal file vuoto congelato')
        else:
            required(config, "configfile", type(None))
            required(config, "configfile_record", type(None))
        if strings(config, "addopts") or strings(config, "effective_args") != [str(repo / p) for p in LEGACY_FILES]:
            raise ValueError("Suite: addopts/selezione effettiva inattesi")
        verbosity = -1 if phase == "collection" else 1
        if (required(config, "verbosity", int) != verbosity or
                required(config, "test_case_verbosity", int) != verbosity):
            raise ValueError("Suite: verbosity effettiva discordante dalla fase")
        if required(config, "cache_dir", str) != str(cache):
            raise ValueError("Suite: cache effettiva diversa da quella attesa")
        for key in ("cache_before", "cache_after"):
            observed_cache = required(data, key, dict)
            if required(observed_cache, "path", str) != str(cache):
                raise ValueError("Suite: path cache discordante dalla fase")
            required(observed_cache, "exists", bool)
            required(observed_cache, "files", list)
            values = required(observed_cache, "values", dict)
            if set(values) != {"nodeids", "lastfailed"}:
                raise ValueError("Suite: osservazioni cache incomplete/inattese")
        ini = required(config, "inicfg", dict).copy()
        if required(ini, "cache_dir", str) != str(cache):
            raise ValueError("Suite: override cache_dir diverso da quello atteso")
        if phase == "run":
            if required(ini, "verbosity_test_cases", str) != "1":
                raise ValueError("Suite: override verbosity_test_cases inatteso")
            del ini["verbosity_test_cases"]
        elif "verbosity_test_cases" in ini:
            raise ValueError("Suite: override verbosity_test_cases non ammesso in collection")
        del ini["cache_dir"]
        if ini:
            raise ValueError("Suite: inicfg comune inatteso per baseline senza config file")
        if not required(config, "pytest_version", str):
            raise ValueError("Suite: versione pytest vuota")
        plugins = required(config, "plugins", list)
        if not plugins:
            raise ValueError("Suite: plugin assenti")
        for plugin in plugins:
            if not required(plugin, "name", str) or not required(plugin, "module", str):
                raise ValueError("Suite: identità plugin incompleta")
        for dist in required(config, "plugin_distributions", list):
            if not required(dist, "name", str) or not required(dist, "version", str):
                raise ValueError("Suite: distribuzione plugin incompleta")
        # I nomi plugin numerici sono ID di oggetti di processi distinti:
        # conservarli nei raw senza imporne l'uguaglianza tra figli.
        common.append({key: config[key] for key in (
            "rootdir", "configfile", "configfile_record", "addopts", "effective_args",
            "pytest_version", "plugin_distributions")} | {"inicfg": ini})
    if common[0] != common[1]:
        raise ValueError("Suite: configurazione/versioni mutate tra collection e run")
    return common[0]


def reconcile_legacy(collection, execution, expected, stdout, cwd, collection_stdout, repo):
    """Riconcilia eventi osservati, summary e cache fresh; mai PASS per complemento."""
    compare_legacy_config(collection, execution, repo, cwd)
    expected_set = unique_nodeids(expected)
    if len(expected_set) != 67 or not set(LEGACY_FAILURES) <= expected_set:
        raise ValueError("Suite: identità baseline attesa incompleta")
    if len(re.findall(r"(?m)^67 tests collected in [0-9.]+s$", collection_stdout)) != 1:
        raise ValueError("Suite: summary collection incoerente/mancante")
    for label, data in (("collection", collection), ("run", execution)):
        if (data["phase"] != label or data["cache_before"]["exists"] or data["cache_before"]["files"] or
                any(v is not None for v in data["cache_before"]["values"].values())):
            raise ValueError("Suite: cache stale/fase errata")
        if unique_nodeids(data["collected"]) != expected_set or data["deselected"] or data["collection_errors"]:
            raise ValueError("Suite: collection difforme/incompleta")
        if data["testscollected"] != 67 or data["exit_code"] != (0 if label == "collection" else 1):
            raise ValueError("Suite: conteggio/exit inatteso")
        if not data.get("config") or not data["config"].get("plugins") or not data["config"].get("pytest_version"):
            raise ValueError("Suite: configurazione/plugin/versioni assenti")
    if collection["cache_before"]["path"] == execution["cache_before"]["path"] or collection["reports"]:
        raise ValueError("Suite: cache non separate o esecuzione durante collection")
    for data in (collection, execution):
        if data.get("session_exit", data["exit_code"]) != data["exit_code"]:
            raise ValueError("Suite: exit sessione difforme")
    # pytest 9.1.1 non salva nodeids durante collect-only (NFPlugin); assenza attesa.
    if any(v not in (None, [], {}) for v in collection["cache_after"]["values"].values()):
        raise ValueError("Suite: collection cache inattesa")
    if unique_nodeids(execution["cache_after"]["values"]["nodeids"]) != expected_set:
        raise ValueError("Suite: nodeids cache run difformi")
    failed_cache = execution["cache_after"]["values"]["lastfailed"]
    if not isinstance(failed_cache, dict) or failed_cache != dict.fromkeys(LEGACY_FAILURES, True):
        raise ValueError("Suite: lastfailed difforme")
    observed = {}
    for item in execution["reports"]:
        node, when = item["nodeid"], item["when"]
        if node not in expected_set or when not in ("setup", "call", "teardown") or (node, when) in observed:
            raise ValueError("Suite: evento mancante/duplicato/ignoto")
        if item.get("wasxfail") or item["outcome"] not in ("passed", "failed"):
            raise ValueError("Suite: skip/xfail/esito inatteso")
        observed[node, when] = item
    outcomes = {}
    cause = "can't open file '" + str(cwd / "clean_markdown.py") + "': [Errno 2] No such file or directory"
    for node in collection["collected"]:
        for when in ("setup", "call", "teardown"):
            item = observed.get((node, when))
            outcome = "failed" if when == "call" and node in LEGACY_FAILURES else "passed"
            if item is None or item["outcome"] != outcome:
                raise ValueError("Suite: esito per-nodeid incompleto/inatteso")
            if outcome == "failed" and (cause not in item["captured"] or "AssertionError" not in item["longrepr"]):
                raise ValueError("Suite: causa lookup legacy non dimostrata")
        outcomes[node] = observed[node, "call"]["outcome"]
    summaries = re.findall(r"(?m)^=*[ \t]*(5 failed, 62 passed(?:, \d+ warnings?)? in [0-9.]+s)[ \t]*=*$", stdout)
    if len(summaries) != 1:
        raise ValueError("Suite: summary incoerente/mancante")
    return {"status": "FAIL", "exit_code": 1, "characterization_complete": True,
            "collected": collection["collected"], "observed_outcomes": outcomes,
            "passed_observed": [n for n, v in outcomes.items() if v == "passed"],
            "failed_observed": [n for n, v in outcomes.items() if v == "failed"],
            "counts": {"collected": 67, "passed": 62, "failed": 5, "skip": 0, "error": 0},
            "summary": summaries[0], "failure_cause": cause}


def file_record(path, root):
    data = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def load_producer():
    spec = importlib.util.spec_from_file_location("a001_source_manifest", Path(__file__).with_name("make_source_manifest.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source_manifest(args):
    tool = load_producer()
    source = json.loads(args.source_manifest.read_text())
    if source.get("schema") != 1 or not source.get("payload"):
        raise ValueError("S v1 mancante/non valido")
    payload = source["payload"]
    if source["id"] != "sha256:" + hashlib.sha256(tool.canonical(payload)).hexdigest():
        raise ValueError("ID di S difforme")
    if payload["repo"] != str(args.modules) or payload["host_repo"] != str(args.repo) or payload["scope"] != "run-stage":
        raise ValueError("Scope/origine baseline incompatibili")
    snapshot = Path(payload["snapshot"]["path"])
    if tool.digest(snapshot) != payload["snapshot"]["sha256"]:
        raise ValueError("Hash snapshot difforme")
    data, indexed = tool.snapshot_check(snapshot, args.repo)
    inventory = payload.get("input_inventory")
    capture_args = SimpleNamespace(repo=args.modules, host_repo=args.repo, snapshot=snapshot,
                                   input_inventory=args.repo / inventory["path"] if inventory else None)
    current = tool.capture(capture_args, data, indexed)
    if current != payload:
        raise ValueError("S non corrisponde agli input correnti; nessun import/prova ammessa")
    runner = json.loads(args.runner_receipt.read_text())
    if runner.get("status") != "PASS" or runner["parent"]["network"]["netns"] != os.readlink("/proc/self/ns/net"):
        raise ValueError("R non PASS nel namespace corrente")
    return source


def command(argv, cwd, name, evidence, expected_exit=0):
    allowed = {'HOME','LANG','LC_ALL','PATH','PIP_CONFIG_FILE','PYTHONDONTWRITEBYTECODE',
               'PYTHONNOUSERSITE','PYTEST_DISABLE_PLUGIN_AUTOLOAD','TZ','TMPDIR','TIKTOKEN_CACHE_DIR',
               'UV_CACHE_DIR','UV_PYTHON_DOWNLOADS','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_CONFIG_DIRS'}
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    if argv[0] == sys.executable:
        if len(argv) > 1 and Path(argv[1]).suffix == '.py':
            script = Path(argv[1])
            argv = [sys.executable, '-I', '-B', str(Path(__file__).with_name('run_trusted_script.py')),
                    '--root', str(script.parent), str(script), *argv[2:]]
        elif argv[1:2] == ['-I'] and argv[2:3] != ['-B']:
            argv = [argv[0], '-I', '-B', *argv[2:]]
    process = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, close_fds=True, timeout=300)
    (evidence / f"{name}-stdout.txt").write_text(process.stdout)
    (evidence / f"{name}-stderr.txt").write_text(process.stderr)
    return {"argv": argv, "cwd": str(cwd), "exit_code": process.returncode, "expected_exit": expected_exit,
            "exit_matches": process.returncode == expected_exit, "stdout": f"{name}-stdout.txt", "stderr": f"{name}-stderr.txt"}


def workspace(args, name):
    root = args.workspaces / name
    root.mkdir(exist_ok=False)
    shutil.copyfile(args.fixtures / "fixture-config.json", root / "pipeline_config.json")
    for directory in ("source_pdfs", "source_documents", "converted_markdown", "cleaned_markdown",
                      "validated_markdown", "chunked_markdown", "pipeline_logs"):
        (root / directory).mkdir()
    return root


def copy_input(source, root, directory):
    shutil.copyfile(source, root / directory / source.name)


def sliding_child(args):
    # Unica aggiunta sys.path, esplicitamente per la copia baseline fidata.
    # Non è usata nelle prove del prodotto installato.
    sys.path.insert(0, str(args.modules))
    from chunk_markdown import MarkdownChunker
    content = (args.fixtures / "f6-multichunk.md").read_text()
    chunker = MarkdownChunker(target_llm="custom", chunk_size=1000, overlap=100)
    encoding = chunker.chunker.token_counter.encoding
    if encoding.name != "cl100k_base" or not content.isascii():
        raise ValueError("Sliding diagnostico limitato a F6 ASCII/cl100k_base")
    # Osserva le chiamate reali del metodo originale senza alterarne i risultati.
    # Il sorgente è tokenizzato una sola volta, dal chunker stesso.
    from unittest.mock import patch
    encoded, decoded = [], []
    original_encode, original_decode = encoding.encode, encoding.decode

    def observe_encode(text, *pos, **kw):
        ids = original_encode(text, *pos, **kw)
        encoded.append((text, list(ids)))
        return ids

    def observe_decode(ids, *pos, **kw):
        text = original_decode(ids, *pos, **kw)
        decoded.append((list(ids), text))
        return text

    with patch.object(encoding, "encode", observe_encode), patch.object(encoding, "decode", observe_decode):
        chunks = chunker.chunk_by_sliding_window(content)
    if len(encoded) != 1 or encoded[0][0] != content:
        raise ValueError("Sliding: tokenizzazione sorgente non unica")
    tokens = encoded[0][1]
    token_bytes = encoding.decode_tokens_bytes(tokens)
    decoded_source, char_offsets = encoding.decode_with_offsets(tokens)
    byte_offsets = [0]
    for data in token_bytes:
        byte_offsets.append(byte_offsets[-1] + len(data))
    if decoded_source != content or char_offsets != byte_offsets[:-1]:
        raise ValueError("Sliding: offsets/decode sorgente difformi")
    windows = [{"start_token": i * 900, "end_token": min(i * 900 + 1000, len(tokens)),
                "source_tokens": ids, "decoded": text} for i, (ids, text) in enumerate(decoded)]
    audit = sliding_audit(content, tokens, token_bytes, windows, chunks)
    intersections = []
    for left, right in zip(chunks, chunks[1:]):
        a, b = encoding.encode(left), encoding.encode(right)
        matches = [n for n in range(1, min(len(a), len(b)) + 1) if a[-n:] == b[:n]]
        intersections.append(max(matches, default=0))
    result = {"encoding": encoding.name, "source_token_count": len(tokens), "chunks": chunks,
              "windows": windows, "observed_suffix_prefix_overlap_tokens": intersections,
              "retokenized_overlap_is_separate_measure_not_gate": True,
              "source_token_ids": tokens, "source_token_byte_offsets": byte_offsets,
              "source_token_char_offsets": char_offsets, "source_encode_calls": 1,
              **audit}
    write_new(args.output, result)
    return 0 if result["multi_chunk"] and result["nonempty_overlap"] else 2


def legacy_suite_child(args):
    check_source_manifest(args)  # Anche nel figlio, prima dell'import di pytest/conftest.
    phase = args.suite_child
    cache = args.workspaces / ("pytest-cache-" + phase)
    before = cache_inventory(cache)
    if before["exists"]:
        raise ValueError("Suite: cache già esistente")
    test_repo = args.modules
    argv = legacy_pytest_args(test_repo, cache, phase)
    data = {"schema": 1, "phase": phase, "argv": argv, "cwd": str(Path.cwd()),
            "cache_before": before, "reports": [], "collected": [], "deselected": [], "collection_errors": []}
    import pytest

    class Observation:
        # Hook osservativi soltanto: nessuna modifica a item/config/esiti.
        def pytest_sessionstart(self, session):
            config = session.config
            self.root = config.rootpath
            config_path = config.inipath
            data["config"] = {
                "rootdir": str(config.rootpath), "configfile": str(config_path) if config_path else None,
                "configfile_record": file_record(config_path, config_path.parent) if config_path else None,
                "inicfg": dict(config.inicfg), "addopts": config.getini("addopts"),
                "effective_args": list(config.args), "invocation_args": list(config.invocation_params.args),
                "verbosity": config.option.verbose, "test_case_verbosity": config.get_verbosity("test_cases"),
                "cache_dir": str(config.cache._cachedir), "pytest_version": pytest.__version__,
                "plugins": [{"name": name, "module": getattr(plugin, "__name__", type(plugin).__module__)}
                            for name, plugin in config.pluginmanager.list_name_plugin() if plugin is not None],
                "plugin_distributions": [{"name": dist.project_name, "version": dist.version}
                                         for _, dist in config.pluginmanager.list_plugin_distinfo()]}

        def nodeid(self, value):
            path, separator, rest = value.partition("::")
            return (self.root / path).resolve().relative_to(test_repo).as_posix() + separator + rest

        def pytest_collection_finish(self, session):
            data["collected"] = [self.nodeid(item.nodeid) for item in session.items]

        def pytest_deselected(self, items):
            data["deselected"].extend(self.nodeid(item.nodeid) for item in items)

        def pytest_collectreport(self, report):
            if report.outcome != "passed":
                data["collection_errors"].append({"nodeid": report.nodeid, "outcome": report.outcome,
                                                  "longrepr": str(report.longrepr)})

        def pytest_runtest_logreport(self, report):
            data["reports"].append({"nodeid": self.nodeid(report.nodeid), "when": report.when,
                                    "outcome": report.outcome, "wasxfail": getattr(report, "wasxfail", None),
                                    "longrepr": str(report.longrepr) if report.failed else "",
                                    "captured": report.capstdout + report.capstderr + report.caplog})

        def pytest_sessionfinish(self, session, exitstatus):
            data["testscollected"] = session.testscollected
            data["session_exit"] = int(exitstatus)

    exit_code = int(pytest.main(argv, plugins=[Observation()]))
    data["exit_code"] = exit_code
    data["cache_after"] = cache_inventory(cache)
    write_new(args.output, data)
    check_source_manifest(args)
    return exit_code


def legacy_suite(args, suite_ws):
    records = {}
    commands = {}
    for phase in ("collection", "run"):
        output = args.output / ("legacy-" + phase + "-observed.json")
        argv = [sys.executable, "-I", str(Path(__file__).resolve()), "--suite-child", phase,
                "--repo", str(args.repo), "--modules", str(args.modules), "--fixtures", str(args.fixtures),
                "--source-manifest", str(args.source_manifest), "--runner-receipt", str(args.runner_receipt),
                "--workspaces", str(suite_ws), "--output", str(output)]
        commands[phase] = command(argv, suite_ws, "legacy-" + phase, args.output, 0 if phase == "collection" else 1)
        records[phase] = json.loads(output.read_text())
        expected = json.loads((args.repo / LEGACY_NODEIDS).read_text())
        if not commands[phase]["exit_matches"] or unique_nodeids(records[phase]["collected"]) != unique_nodeids(expected):
            raise ValueError("Suite: collection/exit inattesi; nessuna caratterizzazione ammessa")
        if phase == "collection" and (records[phase]["deselected"] or records[phase]["collection_errors"] or records[phase]["reports"]):
            raise ValueError("Suite: collection incompleta; esecuzione bloccata")
    audit = reconcile_legacy(records["collection"], records["run"], expected,
                             (args.output / "legacy-run-stdout.txt").read_text(), suite_ws,
                             (args.output / "legacy-collection-stdout.txt").read_text(), args.modules)
    return {**audit, "commands": commands, "observations": {k: "legacy-" + k + "-observed.json" for k in records}}


def run(args):
    source = check_source_manifest(args)  # Prima di qualunque import applicativo/collection.
    args.output.mkdir(exist_ok=False)
    args.workspaces.mkdir(exist_ok=False)
    result = {"schema": 1, "captured_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "baseline-originals",
              "S_id": source["id"], "S_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
              "snapshot": source["payload"]["snapshot"], "runner_receipt": str(args.runner_receipt),
              "netns": os.readlink("/proc/self/ns/net"), "python": sys.executable, "commands": [], "content_checks": [],
              "status": "FAIL", "limitations": []}
    try:
        stable_names = ("f1-stable.md", "f2-sensitive.md", "f5-asset.md", "f6-multichunk.md")
        cleaning = workspace(args, "cleaning")
        (cleaning / "converted_markdown/assets").mkdir()
        shutil.copyfile(args.fixtures / "assets/f5.png", cleaning / "converted_markdown/assets/f5.png")
        for name in stable_names:
            copy_input(args.fixtures / name, cleaning, "converted_markdown")
        result["commands"].append(command([sys.executable, str(args.modules / "clean_markdown.py")], cleaning, "cleaning", args.output))
        for name in stable_names:
            before = (args.fixtures / name).read_bytes()
            after = (cleaning / "cleaned_markdown" / name).read_bytes()
            diff = "".join(difflib.unified_diff(before.decode().splitlines(keepends=True), after.decode().splitlines(keepends=True),
                                             fromfile="source/" + name, tofile="cleaned/" + name))
            (args.output / f"{name}-cleaning.diff").write_text(diff)
            result["content_checks"].append({"fixture": name, "cleaned_bytes": len(after), "identical_to_source": before == after,
                                              "diff": f"{name}-cleaning.diff", "nonempty": bool(after.strip())})
        result["limitations"].append({"F5_asset_copied_by_cleaner": (cleaning / "cleaned_markdown/assets/f5.png").exists(),
                                       "note": "Asset separato inventariato; nessuna riparazione/bundle completo dichiarato."})
        for label, input_dir in (("validation-chain", cleaning / "cleaned_markdown"), ("f3-validation", args.fixtures)):
            ws = workspace(args, label)
            paths = sorted(input_dir.glob("*.md")) if label == "validation-chain" else [args.fixtures / "f3-cleaned-stable.md", args.fixtures / "f3-cleaned-sensitive.md"]
            for path in paths:
                copy_input(path, ws, "cleaned_markdown")
            result["commands"].append(command([sys.executable, str(args.modules / "validate_markdown.py")], ws, label, args.output))
            report = json.loads((ws / "validation_report.json").read_text())
            result["content_checks"].append({"case": label, "report": report, "copied_files_byte_identical": all(
                p.read_bytes() == (ws / "cleaned_markdown" / p.name).read_bytes() for p in (ws / "validated_markdown").glob("*.md"))})
        f4 = workspace(args, "f4-conversion")
        copy_input(args.fixtures / "f4-pandoc.html", f4, "source_documents")
        argv = [sys.executable, str(args.modules / "unified_converter.py"), "--source-dir", str(f4 / "source_documents"),
                "--output-dir", str(f4 / "converted_markdown"), "--max-workers", "1", "--force"]
        result["commands"].append(command(argv, f4, "f4-conversion", args.output))
        summary = json.loads((f4 / "converted_markdown/unified_conversion_summary.json").read_text())
        text = (f4 / "converted_markdown/f4-pandoc.md").read_text()
        invariants = ["123,45", "-7", "2026", "Città", "perché", "quantità", "E = mc²"]
        references = f4_references((args.fixtures / "f4-pandoc.html").read_text(), text)
        result["content_checks"].append({"case": "F4", "summary": summary, "invariants": {v: v in text for v in invariants},
                                          "references": references,
                                          "table": bool(re.search(r"(?m)^\s*Caso\s+Valore\s*\n\s*-+\s+-+\s*\n\s*A\s+123,45\s*\n\s*B\s+-7\s*$", text)),
                                          "section_order": "Primo esperimento" in text and "Secondo esperimento" in text and
                                          text.index("Primo esperimento") < text.index("Secondo esperimento")})
        result["limitations"].append({"F4_HTML_anchor_omitted": 'id="riferimento"' not in text and "{#riferimento}" not in text,
                                     "note": "Perdita legacy dell'anchor HTML; riferimenti testuali verificati separatamente."})
        for size, overlap in ((1000, 100), (1200, 80), (1600, 120)):
            label = f"f6-chunk-{size}-{overlap}"
            ws = workspace(args, label)
            copy_input(args.fixtures / "f6-multichunk.md", ws, "validated_markdown")
            argv = [sys.executable, str(args.modules / "chunk_markdown.py"), "--input-dir", str(ws / "validated_markdown"),
                    "--output-dir", str(ws / "chunked_markdown"), "--target-llm", "custom", "--chunk-size", str(size), "--overlap", str(overlap)]
            result["commands"].append(command(argv, ws, label, args.output))
            metadata = json.loads((ws / "chunked_markdown/chunking_metadata.json").read_text())
            chunks = sorted((ws / "chunked_markdown").glob("f6-multichunk_chunk_*.md"))
            result["content_checks"].append({"case": label, "chunk_count": len(chunks), "multi_chunk": len(chunks) >= 2,
                                              "metadata": metadata, "chunk_files": [file_record(p, ws) for p in chunks]})
        slide_cwd = workspace(args, "f6-sliding")
        argv = [sys.executable, "-I", str(Path(__file__).resolve()), "--sliding-child", "--modules", str(args.modules),
                "--fixtures", str(args.fixtures), "--output", str(slide_cwd / "sliding.json")]
        result["commands"].append(command(argv, slide_cwd, "f6-sliding", args.output))
        result["content_checks"].append({"case": "F6-sliding", "output": str(slide_cwd / "sliding.json")})
        negative = workspace(args, "legacy-external-orchestrator")
        copy_input(args.fixtures / "f1-stable.md", negative, "converted_markdown")
        result["commands"].append(command([sys.executable, str(args.modules / "master_workflow.py"), "--step", "cleaning", "--force"],
                                          negative, "legacy-external-orchestrator", args.output, expected_exit=1))
        result["limitations"].append("Orchestratore originale da CWD esterno: difetto di ricerca degli script, exit 1 atteso.")
        config_ws = workspace(args, "config-cli")
        for option in ("--create-default", "--show", "--validate"):
            result["commands"].append(command([sys.executable, str(args.modules / "config.py"), option], config_ws,
                                              "config-" + option[2:], args.output))
        # La suite legacy può mostrare il difetto dei figli originali. Esito distinto
        # dal confronto contenuti; nessun successo attribuito a una suite fallita.
        suite_ws = workspace(args, "legacy-suite")
        result["legacy_suite"] = legacy_suite(args, suite_ws)
        result["workspaces"] = [file_record(p, args.workspaces) for p in sorted(args.workspaces.rglob("*")) if p.is_file()]
        checks_ok = all(c["exit_matches"] for c in result["commands"])
        checks_ok = checks_ok and all(item.get("multi_chunk", True) for item in result["content_checks"])
        checks_ok = checks_ok and all(item.get("nonempty", True) for item in result["content_checks"])
        checks_ok = checks_ok and all(item.get("copied_files_byte_identical", True) for item in result["content_checks"])
        f4check = next(item for item in result["content_checks"] if item.get("case") == "F4")
        checks_ok = checks_ok and all(f4check["invariants"].values()) and f4check["section_order"] and f4check["table"]
        checks_ok = checks_ok and f4check["summary"]["conversion_summary"]["failed"] == 0
        check_source_manifest(args)
        result["status"] = "PASS" if checks_ok else "FAIL"
        result["scope_of_PASS"] = "Caratterizzazione contenuti baseline; legacy_suite conserva il proprio exit, non è promossa a PASS."
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    write_new(args.output / "receipt.json", result)
    print(json.dumps({"status": result["status"], "receipt": str(args.output / "receipt.json")}))
    return 0 if result["status"] == "PASS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--runner-receipt", type=Path)
    parser.add_argument("--workspaces", type=Path)
    parser.add_argument("--sliding-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--suite-child", choices=("collection", "run"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    for path in (args.modules, args.fixtures, args.output, args.repo, args.source_manifest, args.runner_receipt, args.workspaces):
        if path is not None and (not path.is_absolute() or any(part.is_symlink() for part in (path, *path.parents))):
            parser.error("Sono richiesti path assoluti senza symlink")
    if args.sliding_child:
        return sliding_child(args)
    if any(value is None for value in (args.repo, args.source_manifest, args.runner_receipt, args.workspaces)):
        parser.error("V0 richiede repo, S, R e workspaces")
    if args.suite_child:
        return legacy_suite_child(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
