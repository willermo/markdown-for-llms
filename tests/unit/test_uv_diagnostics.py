"""Controlli stdlib dei gate; mock/sintetici, non costituiscono R o V0."""

import importlib.util
import copy
import json
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[2]


def diagnostic(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/diagnostics/run-a001-fase0-uv" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDiagnosticGates(unittest.TestCase):
    def test_netlink_attributes_reject_truncation_and_invalid_lengths(self):
        runner = diagnostic("check_runner")
        valid = struct.pack("=HH", 7, 3) + b"lo\0" + b"\0"
        self.assertEqual(runner.attributes(valid), {3: b"lo\0"})
        for data in (b"\0", struct.pack("=HH", 2, 3), struct.pack("=HH", 8, 3) + b"ab"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                runner.attributes(data)

    def test_ip_probes_are_never_attempted_in_host_or_routed_namespace(self):
        runner = diagnostic("check_runner")
        target = {"daemons": [], "tmpdir": "/tmp", "tokenizer_cache": "/tmp", "binaries": {}}
        for namespace, safe in (("net:[host]", True), ("net:[isolated]", False)):
            left, right = MagicMock(), MagicMock()
            right.recv.return_value = b"a001-local"
            with self.subTest(namespace=namespace, safe=safe), \
                 patch.object(runner, "network_state", return_value={"netns": namespace, "no_external_routes_or_addresses": safe}), \
                 patch.object(runner, "prerequisites", return_value=[{"matches": False}]), \
                 patch.object(runner.socket, "socketpair", return_value=(left, right)), \
                 patch.object(runner, "connect_only", return_value={"connected": False, "errno": 2}) as connect:
                value = runner.observe(target, "net:[host]", "/tmp/synthetic-only.sock")
                self.assertEqual(value["status"], "IMPEDITA")
                self.assertEqual(value["internet_probes"], [])
                self.assertTrue(value["socketpair_positive"])
                self.assertEqual([call.args[0] for call in connect.call_args_list], [socket.AF_UNIX])

    def test_denied_socketpair_keeps_runner_impeded(self):
        runner = diagnostic("check_runner")
        with patch.object(runner, "network_state", return_value={"netns": "net:[host]", "no_external_routes_or_addresses": True}), \
             patch.object(runner.socket, "socketpair", side_effect=PermissionError("synthetic policy denial")):
            value = runner.observe({"daemons": []}, "net:[host]", "/tmp/synthetic.sock")
        self.assertEqual(value["status"], "IMPEDITA")
        self.assertEqual(value["internet_probes"], [])
        self.assertIn("PermissionError", value["error"])

    def test_routes_are_loopback_or_rejection_never_external_default(self):
        runner = diagnostic("check_runner")

        def attr(kind, data):
            item = struct.pack("=HH", len(data) + 4, kind) + data
            return item + b"\0" * (-len(item) % 4)

        link = struct.pack("=BBHiII", 0, 0, 772, 1, 1, 0) + attr(3, b"lo\0")
        proc = {"/proc/net/dev": "head\nhead\n lo: 0\n", "/proc/net/route": "head\n",
                "/proc/net/ipv6_route": "", "/proc/net/if_inet6": ""}
        for typ, prefix, destination, expected in ((2, 8, "127.0.0.0", True), (1, 0, None, False),
                                                   (7, 0, None, True), (1, 0, "127.0.0.0", False)):
            route = struct.pack("=BBBBBBBBI", socket.AF_INET, prefix, 0, 0, 254, 2, 254, typ, 0) + attr(4, struct.pack("=I", 1))
            if destination:
                route += attr(1, socket.inet_aton(destination))
            with self.subTest(typ=typ, prefix=prefix), \
                 patch.object(runner, "netlink_dump", side_effect=[[(16, link)], [], [(24, route)]]), \
                 patch.object(runner.Path, "read_text", autospec=True, side_effect=lambda path: proc[str(path)]), \
                 patch.object(runner.os, "readlink", return_value="net:[synthetic]"):
                self.assertEqual(runner.network_state()["no_external_routes_or_addresses"], expected)

    def test_source_manifest_rejects_symlink_and_snapshot_hash_mismatch(self):
        tool = diagnostic("make_source_manifest")
        with tempfile.TemporaryDirectory(prefix="a001-unit-") as directory:
            root = Path(directory)
            file = root / "module.py"
            file.write_bytes(b"synthetic = 1\n")
            link = root / "link.py"
            link.symlink_to(file)
            with self.assertRaises(ValueError):
                tool.entry(link, root)
            item = tool.entry(file, root)
            with self.assertRaises(ValueError):
                tool.match_snapshot([item], {item["path"]: {"kind": "file", "sha256": "0" * 64}})
            with self.assertRaises(ValueError):
                tool.match_snapshot([item], {})

    def test_source_id_uses_canonical_payload_independent_of_key_order(self):
        tool = diagnostic("make_source_manifest")
        self.assertEqual(tool.canonical({"b": 2, "a": "città"}), tool.canonical({"a": "città", "b": 2}))
        self.assertNotEqual(tool.canonical({"a": 1}), tool.canonical({"a": 2}))

    def test_stale_stage_stops_before_socket_or_command(self):
        wrapper = diagnostic("run_offline")
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(prefix="a001-unit-") as directory:
            output = Path(directory) / "receipt"
            args = SimpleNamespace(output=output, runner="unshare", target=ROOT / "AGENTS.md",
                                   repo=ROOT, snapshot=ROOT / "AGENTS.md", python="/synthetic/python")
            with patch.object(wrapper, "snapshot_check", side_effect=ValueError("STALE")), \
                 patch.object(wrapper.tempfile, "mkdtemp") as new_socket, \
                 patch.object(wrapper.subprocess, "run") as execute:
                self.assertEqual(wrapper.outside(args, {"daemons": []}), 2)
                new_socket.assert_not_called()
                execute.assert_not_called()
            self.assertEqual(json.loads((output / "receipt.json").read_text())["status"], "FAIL")


class TestBaselineF4(unittest.TestCase):
    def setUp(self):
        self.tool = diagnostic("run_baseline")
        self.source = (ROOT / "tests/fixtures/run-a001-fase0-uv/f4-pandoc.html").read_text()
        self.text = ("# Primo esperimento\n\nIl riferimento \\[1\\] segue il primo risultato.\n\n"
                     "## Secondo esperimento\n\n\\[1\\] Autore sintetico, Titolo, 2026.\n")

    def test_scoped_escaped_and_literal_references(self):
        for text in (self.text, self.text.replace("\\[1\\]", "[1]")):
            self.assertEqual(self.tool.f4_references(self.source, text)["output_count"], 2)

    def test_missing_changed_duplicated_or_moved_reference_rejected(self):
        for text in (self.text.replace("\\[1\\]", "", 1), self.text.replace("\\[1\\]", "\\[2\\]", 1),
                     self.text + "\\[1\\]\n", self.text + "[2]\n", self.text.replace("Il riferimento \\[1\\]", "Il riferimento") + "\\[1\\]\n",
                     self.text.replace("Autore sintetico", "Altro autore")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.tool.f4_references(self.source, text)

    def test_sections_exist_and_reference_stays_in_first_section(self):
        for text in (self.text.replace("Primo esperimento", ""), self.text.replace("Secondo esperimento", ""),
                     self.text.replace("# Primo esperimento", "## Secondo esperimento", 1).replace("## Secondo esperimento\n\n\\[", "# Primo esperimento\n\n\\["),
                     "# Primo esperimento\n## Secondo esperimento\nIl riferimento [1] segue il primo risultato.\n[1] Autore sintetico, Titolo, 2026.\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.tool.f4_references(self.source, text)


class TestBaselineSliding(unittest.TestCase):
    def setUp(self):
        self.tool = diagnostic("run_baseline")
        self.content = "abcdefghijklmnop\n"
        self.tokens = list(range(len(self.content)))
        self.parts = [c.encode("ascii") for c in self.content]
        self.windows = [{"start_token": a, "end_token": b, "source_tokens": self.tokens[a:b],
                         "decoded": self.content[a:b]} for a, b in ((0, 8), (6, 14), (12, 17))]
        self.chunks = [w["decoded"].strip() for w in self.windows]

    def audit(self, **changes):
        data = dict(content=self.content, tokens=self.tokens, token_bytes=self.parts,
                    windows=self.windows, chunks=self.chunks, size=8, overlap=2)
        data.update(changes)
        return self.tool.sliding_audit(**data)

    def test_exact_windows_shared_text_and_trimmed_final_margin(self):
        audit = self.audit()
        self.assertEqual([i["shared_text"] for i in audit["intersections"]], ["gh", "mn"])
        self.assertEqual(audit["excluded_trailing"], "\n")
        self.assertTrue(audit["continuous_coverage"])
        self.assertEqual(audit["source_end_byte"], 17)

    def test_changed_window_id_or_intersection_rejected(self):
        for index in (0, 6):
            windows = copy.deepcopy(self.windows)
            windows[0]["source_tokens"][index] = 999
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.audit(windows=windows)

    def test_changed_body_decoded_text_or_token_bytes_rejected(self):
        chunks = self.chunks.copy(); chunks[1] = "X" + chunks[1][1:]
        windows = copy.deepcopy(self.windows); windows[0]["decoded"] = "changed"
        parts = self.parts.copy(); parts[0] = b"X"
        for changes in ({"chunks": chunks}, {"windows": windows}, {"token_bytes": parts}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.audit(**changes)

    def test_gap_order_duplicate_truncated_tail_and_wrong_parameters_rejected(self):
        windows = copy.deepcopy(self.windows); windows[1]["start_token"] += 1
        for changes in ({"windows": windows}, {"windows": self.windows[::-1], "chunks": self.chunks[::-1]},
                        {"windows": self.windows + self.windows[-1:], "chunks": self.chunks + self.chunks[-1:]},
                        {"windows": self.windows[:-1], "chunks": self.chunks[:-1]}, {"overlap": 0},
                        {"chunks": [self.chunks[0], self.chunks[1], self.chunks[2][:-1]]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.audit(**changes)

    def test_zero_observable_text_and_non_ascii_rejected(self):
        content = "ab  cd"
        windows = [{"start_token": 0, "end_token": 4, "source_tokens": [0, 1, 2, 3]},
                   {"start_token": 2, "end_token": 6, "source_tokens": [2, 3, 4, 5]}]
        with self.assertRaises(ValueError):
            self.audit(content=content, tokens=list(range(6)), token_bytes=[c.encode() for c in content],
                       windows=windows, chunks=["ab", "cd"], size=4, overlap=2)
        with self.assertRaises(ValueError):
            self.audit(content=self.content + "é")


class TestBaselineSuite(unittest.TestCase):
    def setUp(self):
        self.tool = diagnostic("run_baseline")
        self.nodes = ["tests/unit/test_cleaning.py::test_synthetic_" + str(i) for i in range(62)] + list(self.tool.LEGACY_FAILURES)
        self.repo = Path("/synthetic/repo")
        self.cwd = Path("/synthetic/legacy-suite")
        self.summary = "===== 5 failed, 62 passed in 2.16s =====\n"
        self.collection_summary = "67 tests collected in 0.08s\n"
        self.collection = dict(phase="collection", collected=self.nodes, deselected=[], collection_errors=[],
                               testscollected=67, exit_code=0, reports=[])
        self.execution = copy.deepcopy(self.collection)
        self.execution.update(phase="run", exit_code=1)
        selected = ["/synthetic/repo/tests/unit/test_cleaning.py", "/synthetic/repo/tests/unit/test_validation.py",
                    "/synthetic/repo/tests/unit/test_chunking.py", "/synthetic/repo/tests/integration/test_pipeline.py"]
        for phase, data in (("collection", self.collection), ("run", self.execution)):
            cache = "/synthetic/legacy-suite/pytest-cache-" + phase
            display = ["--collect-only", "-q", "--verbosity=-1"] if phase == "collection" else [
                "-v", "-rA", "--verbosity=1", "-o", "verbosity_test_cases=1"]
            argv = selected + display + ["-o", "cache_dir=" + cache]
            data.update(schema=1, cwd=str(self.cwd), argv=argv, session_exit=data["exit_code"])
            data["config"] = dict(rootdir=str(self.repo), configfile=None, configfile_record=None,
                inicfg={"cache_dir": cache} if phase == "collection" else {"cache_dir": cache, "verbosity_test_cases": "1"},
                addopts=[], effective_args=selected.copy(), invocation_args=argv.copy(),
                verbosity=-1 if phase == "collection" else 1, test_case_verbosity=-1 if phase == "collection" else 1,
                cache_dir=cache, pytest_version="9.1.1",
                plugins=[{"name": "123" if phase == "collection" else "456", "module": "_pytest.config"},
                         {"name": "pytest_mock", "module": "pytest_mock"}],
                plugin_distributions=[{"name": "pytest-mock", "version": "3.16.0"}, {"name": "pytest-cov", "version": "7.1.0"}])
            data["cache_before"] = dict(path=cache, exists=False, files=[], values={"nodeids": None, "lastfailed": None})
            data["cache_after"] = copy.deepcopy(data["cache_before"])
        self.execution["cache_after"]["exists"] = True
        self.execution["cache_after"]["files"] = [dict(path="v/cache/nodeids", bytes=100, sha256="a" * 64)]
        self.execution["cache_after"]["values"] = {"nodeids": self.nodes, "lastfailed": dict.fromkeys(self.tool.LEGACY_FAILURES, True)}
        for node in self.nodes:
            for when in ("setup", "call", "teardown"):
                failed = node in self.tool.LEGACY_FAILURES and when == "call"
                self.execution["reports"].append(dict(nodeid=node, when=when, outcome="failed" if failed else "passed",
                    wasxfail=None, captured="can't open file '/synthetic/legacy-suite/clean_markdown.py': [Errno 2] No such file or directory" if failed else "",
                    longrepr="AssertionError" if failed else ""))

    def reconcile(self):
        return self.tool.reconcile_legacy(self.collection, self.execution, self.nodes, self.summary, self.cwd, self.collection_summary, self.repo)

    def test_observed_passes_and_failures_reconcile_without_promoting_suite(self):
        result = self.reconcile()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(len(result["passed_observed"]), 62)
        self.assertEqual(len(result["failed_observed"]), 5)

    def test_missing_duplicate_unknown_or_skipped_event_rejected(self):
        original = copy.deepcopy(self.execution["reports"])
        for mode in ("missing", "duplicate", "unknown", "skip", "xfail"):
            self.execution["reports"] = copy.deepcopy(original)
            if mode == "missing": self.execution["reports"].pop()
            if mode == "duplicate": self.execution["reports"].append(self.execution["reports"][0])
            if mode == "unknown": self.execution["reports"][0]["nodeid"] = "unknown"
            if mode == "skip": self.execution["reports"][0]["outcome"] = "skipped"
            if mode == "xfail": self.execution["reports"][0]["wasxfail"] = "synthetic"
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.reconcile()

    def test_stale_cache_or_missing_identity_rejected(self):
        original = copy.deepcopy(self.execution)
        for mode in ("stale", "same", "nodeids", "lastfailed", "duplicate"):
            self.execution = copy.deepcopy(original)
            if mode == "stale": self.execution["cache_before"]["exists"] = True
            if mode == "same": self.execution["cache_before"]["path"] = self.collection["cache_before"]["path"]
            if mode == "nodeids": self.execution["cache_after"]["values"]["nodeids"].pop()
            if mode == "lastfailed": self.execution["cache_after"]["values"]["lastfailed"] = {}
            if mode == "duplicate": self.execution["collected"] = self.nodes + self.nodes[:1]
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.reconcile()

    def test_wrong_count_summary_exit_failure_cause_or_deselection_rejected(self):
        original = copy.deepcopy(self.execution)
        for mode in ("count", "summary", "exit", "cause", "deselected", "error", "config"):
            self.execution = copy.deepcopy(original); self.summary = "5 failed, 62 passed in 2.16s\n"
            if mode == "count": self.execution["testscollected"] = 68
            if mode == "summary": self.summary = "4 failed, 63 passed in 2.16s\n"
            if mode == "exit": self.execution["exit_code"] = 0
            if mode == "cause": self.execution["reports"][-2]["captured"] = "other error"
            if mode == "deselected": self.execution["deselected"] = self.nodes[:1]
            if mode == "error": self.execution["collection_errors"] = ["error"]
            if mode == "config": self.execution["config"] = {}
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.reconcile()

    def test_pytest_selection_and_separate_absolute_caches(self):
        for phase in ("collection", "run"):
            cache = Path("/synthetic") / phase
            argv = self.tool.legacy_pytest_args(ROOT, cache, phase)
            self.assertEqual(argv[:4], [str(ROOT / name) for name in self.tool.LEGACY_FILES])
            self.assertIn("cache_dir=" + str(cache), argv)
            self.assertFalse(any("test_uv_diagnostics" in item for item in argv))
        with self.assertRaises(ValueError): self.tool.legacy_pytest_args(ROOT, Path("relative"), "run")

    def test_collection_summary_and_session_exit_must_match(self):
        self.collection_summary = "66 tests collected in 0.08s\n"
        with self.assertRaises(ValueError): self.reconcile()
        self.collection_summary = "67 tests collected in 0.08s\n"
        self.execution["session_exit"] = 0
        with self.assertRaises(ValueError): self.reconcile()

    def test_exact_phase_overrides_preserve_raw_and_common_config(self):
        before = copy.deepcopy((self.collection, self.execution))
        common = self.tool.compare_legacy_config(self.collection, self.execution, self.repo, self.cwd)
        self.assertEqual(common["inicfg"], {})
        self.assertIsNone(common["configfile"])
        self.assertIsNone(common["configfile_record"])
        self.assertEqual(common["addopts"], [])
        self.assertEqual(self.reconcile()["exit_code"], 1)
        self.assertEqual((self.collection, self.execution), before)

    def test_required_config_fields_missing_null_or_wrong_type_rejected(self):
        originals = copy.deepcopy((self.collection, self.execution))
        for phase in (0, 1):
            for key in originals[phase]["config"]:
                for mode in ("missing", "null", "wrong_type"):
                    if mode == "null" and key in ("configfile", "configfile_record"):
                        continue  # Null esplicito atteso; la mancanza è testata.
                    self.collection, self.execution = copy.deepcopy(originals)
                    config = (self.collection, self.execution)[phase]["config"]
                    if mode == "missing": del config[key]
                    else: config[key] = None if mode == "null" else False
                    with self.subTest(phase=phase, key=key, mode=mode), self.assertRaises(ValueError):
                        self.reconcile()
        # Entrambi mancanti non devono essere accettati da get()==get().
        for key in originals[0]["config"]:
            self.collection, self.execution = copy.deepcopy(originals)
            del self.collection["config"][key]; del self.execution["config"][key]
            with self.subTest(both_missing=key), self.assertRaises(ValueError): self.reconcile()

    def test_required_invocation_and_cache_fields_rejected(self):
        originals = copy.deepcopy((self.collection, self.execution))
        paths = [(key,) for key in ("phase", "cwd", "argv", "config", "cache_before", "cache_after")]
        paths += [(cache, key) for cache in ("cache_before", "cache_after") for key in ("path", "exists", "files", "values")]
        paths += [(cache, "values", key) for cache in ("cache_before", "cache_after") for key in ("nodeids", "lastfailed")]
        for phase in (0, 1):
            for path in paths:
                for mode in ("missing", "wrong_type"):
                    if len(path) == 3 and mode == "wrong_type": continue
                    self.collection, self.execution = copy.deepcopy(originals)
                    value = (self.collection, self.execution)[phase]
                    for key in path[:-1]: value = value[key]
                    if mode == "missing": del value[path[-1]]
                    else: value[path[-1]] = None
                    with self.subTest(phase=phase, path=path, mode=mode), self.assertRaises(ValueError): self.reconcile()

    def test_missing_extra_wrong_or_duplicate_overrides_rejected(self):
        originals = copy.deepcopy((self.collection, self.execution))
        for phase in (0, 1):
            for mode in ("cache_missing", "cache_wrong", "extra_ini", "verbosity_missing", "verbosity_wrong",
                         "extra_argv", "duplicate_cache", "duplicate_verbosity", "missing_argv", "wrong_argv", "display"):
                self.collection, self.execution = copy.deepcopy(originals)
                data = (self.collection, self.execution)[phase]; ini = data["config"]["inicfg"]
                if mode == "cache_missing": del ini["cache_dir"]
                if mode == "cache_wrong": ini["cache_dir"] = False
                if mode == "extra_ini": ini["addopts"] = "-k other"
                if mode == "verbosity_missing":
                    if phase: del ini["verbosity_test_cases"]
                    else: ini["verbosity_test_cases"] = "-1"
                if mode == "verbosity_wrong": ini["verbosity_test_cases"] = 1
                if mode == "extra_argv": data["argv"] += ["-o", "addopts=-k other"]
                if mode == "duplicate_cache": data["argv"] += ["-o", "cache_dir=" + data["config"]["cache_dir"]]
                if mode == "duplicate_verbosity": data["argv"] += ["-o", "verbosity_test_cases=1"]
                if mode == "missing_argv": data["argv"] = data["argv"][:-2]
                if mode == "wrong_argv": data["argv"][-1] = "cache_dir=/other"
                if mode == "display": data["argv"][4] = "-qq"
                # Anche argv e invocation concordanti non rendono valido un extra.
                data["config"]["invocation_args"] = data["argv"].copy()
                with self.subTest(phase=phase, mode=mode), self.assertRaises(ValueError): self.reconcile()

    def test_cache_phase_paths_cannot_self_authorize(self):
        originals = copy.deepcopy((self.collection, self.execution))
        for phase in (0, 1):
            for wrong in ("/external/cache", "/synthetic/old/legacy-suite/pytest-cache-run",
                          originals[1-phase]["config"]["cache_dir"], "relative/cache"):
                self.collection, self.execution = copy.deepcopy(originals)
                data = (self.collection, self.execution)[phase]
                data["config"]["cache_dir"] = data["config"]["inicfg"]["cache_dir"] = wrong
                data["cache_before"]["path"] = data["cache_after"]["path"] = wrong
                data["argv"][-1] = "cache_dir=" + wrong
                data["config"]["invocation_args"] = data["argv"].copy()
                with self.subTest(phase=phase, wrong=wrong), self.assertRaises(ValueError): self.reconcile()
        self.collection, self.execution = copy.deepcopy(originals)
        self.collection, self.execution = self.execution, self.collection
        with self.assertRaises(ValueError): self.reconcile()

    def test_common_mutations_not_hidden_by_different_caches(self):
        originals = copy.deepcopy((self.collection, self.execution))
        mutations = {"rootdir": "/other", "configfile": "/synthetic/pytest.ini",
                     "configfile_record": {"path": "pytest.ini", "bytes": 12, "sha256": "b" * 64},
                     "addopts": ["-k", "other"], "effective_args": ["other.py"],
                     "inicfg": {"cache_dir": "/synthetic/legacy-suite/pytest-cache-run", "unexpected": "value"},
                     "pytest_version": "9.2.0", "plugin_distributions": [{"name": "pytest-mock", "version": "0.0"}]}
        for key, value in mutations.items():
            self.collection, self.execution = copy.deepcopy(originals)
            self.execution["config"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.reconcile()
        for key in ("addopts", "configfile", "configfile_record", "effective_args"):
            self.collection, self.execution = copy.deepcopy(originals)
            for data in (self.collection, self.execution): data["config"][key] = mutations[key]
            with self.subTest(both_mutated=key), self.assertRaises(ValueError): self.reconcile()
        self.collection, self.execution = copy.deepcopy(originals)
        for data in (self.collection, self.execution): data["config"]["inicfg"]["unexpected"] = "same"
        with self.assertRaises(ValueError): self.reconcile()

    def test_effective_config_must_match_invocation(self):
        originals = copy.deepcopy((self.collection, self.execution))
        for phase in (0, 1):
            for path, value in ((('cwd',), '/other/legacy-suite'), (('phase',), 'other'),
                                (('config', 'invocation_args'), []), (('config', 'verbosity'), 0),
                                (('config', 'test_case_verbosity'), 0), (('config', 'cache_dir'), '/other'),
                                (('cache_before', 'path'), '/other'), (('cache_after', 'path'), '/other')):
                self.collection, self.execution = copy.deepcopy(originals)
                data = (self.collection, self.execution)[phase]
                target = data if len(path) == 1 else data[path[0]]
                target[path[-1]] = value
                with self.subTest(phase=phase, path=path), self.assertRaises(ValueError): self.reconcile()

    def test_nested_plugin_and_selection_types_are_checked(self):
        original = copy.deepcopy(self.execution)
        for key, value in (("plugins", []), ("plugins", [None]), ("plugins", [{"name": "id"}]),
                           ("plugins", [{"name": 123, "module": "plugin"}]),
                           ("plugin_distributions", [None]), ("plugin_distributions", [{"name": "plugin"}]),
                           ("plugin_distributions", [{"name": "plugin", "version": 1}]),
                           ("pytest_version", ""), ("effective_args", [None]), ("addopts", [1]),
                           ("invocation_args", [None])):
            self.execution = copy.deepcopy(original); self.execution["config"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.reconcile()

    def test_expected_paths_must_be_explicit_absolute_parameters(self):
        for repo, cwd in ((Path("relative"), self.cwd), (self.repo, Path("relative")),
                          (self.repo, Path("/synthetic/other")), (Path("/synthetic/../repo"), self.cwd)):
            with self.subTest(repo=repo, cwd=cwd), self.assertRaises(ValueError):
                self.tool.compare_legacy_config(self.collection, self.execution, repo, cwd)


if __name__ == "__main__":
    unittest.main()
