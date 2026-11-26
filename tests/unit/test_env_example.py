"""`.env.example` is complete (MFL033 N6; the maintainer, 18/11): every environment variable that the
Python code of the product and of `scripts/` reads appears in it, as a setting or as a commented line,
or is excluded below with its reason; the keys appear by name only, commented and without a value.

What "reads" means here (an AST scan of `markdown_for_llms/` and `scripts/`, never executed): a call
`get`, `pop` or `getenv` with a literal name on an object named `environ`, `env` or `environment`
(`os.environ.get("X")`, `self.environ.get("X")`, `env.get("X")`, `os.getenv("X")`), a subscript
`environ["X"]` that loads, a test `"X" in environ`, and the names assigned to a module constant whose
name ends in `_ENV` (`LOG_ENV = "SESSION_WORKER_LOG"`); then every field of the product settings
(`MDF_<FIELD>`), of the bench settings (`MDF_BENCH_<FIELD>`) and the names of the keys. The variables
of the shell scripts (`FULL_CHECK_*`, `RUN_*` of `scripts/full-check.sh`) are documented in
`docs/how-to/uv-environment.md` and are not in this scan.

A read whose name is not a literal (S-NB3 of the final review of MFL033: `env.get("MDF_" + key.upper())`)
passes only when it is in DYNAMIC, a closed table of each such read by module and expression, with the
names it can read resolved from the code (a constant, the rules of the watchdog); a new one fails the
scan until it is listed, and every name it resolves to is checked like a literal one.
"""
import ast
import dataclasses
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ("markdown_for_llms", "scripts")
ENVISH = {"environ", "env", "environment"}
NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
LINE = re.compile(r"^(#\s*)?([A-Z_][A-Z0-9_]*)=(.*)$", re.M)

# Read by the code and kept out of `.env.example` on purpose, each with its reason.
EXCLUDED = {
    # the system and the session of the user: not settings of this project
    "PATH": "the search path of the processes",
    "PYTHONPATH": "the import path of a child process (the watchdog, a legacy diagnostic)",
    "XDG_CACHE_HOME": "the cache base directory of the user, for the default model, CPU server and bench cache roots",
    "XDG_CONFIG_HOME": "the configuration base directory of the user, for the default servers.json",
    "XDG_DATA_HOME": "the data base directory of the user, for the default runs folder of the bench",
    # set by scripts/servers/session.py for the command it wraps, never by the user
    "MDF_SESSION_ARM": "the window of a live session, set by session.py run for its command",
    "SESSION_WORKER_LOG": "the log of the workers of a live session, set by session.py run for its command",
    # the environment of the remote container of a RunPod endpoint or a Modal app, set by scripts/servers at the
    # arm or in the image, never by the user and never on this machine
    "MDF_PROVIDER": "the container of the endpoint (watchdog.py): its provider, set by the endpoint or the app",
    "MDF_ALLOW_DOWNLOAD": "the container of the endpoint: the populate of the weights, set by runpod_endpoint.py",
    "MDF_FAULT": "the container of the endpoint: the fault of a fault test, set by session.py at the arm",
    "MDF_LEVERS": "the container of the endpoint: the levers of the start, set by session.py at the arm",
    "MDF_MODAL_APP": "the name of the Modal app at its deploy, set by session.py",
    "MDF_UPSTREAM_PORT": "the container of the endpoint: the port of vLLM behind the watchdog",
    "MDF_WATCHDOG_B64": "the container of the RunPod endpoint: the watchdog itself, set by runpod_endpoint.py",
    "MDF_WATCHDOG_SHA256": "the container of the RunPod endpoint: the SHA-256 of the watchdog",
    "PORT": "the container of the endpoint: the port of the watchdog's proxy",
    "HF_HOME": "the container of the endpoint: the weights on the volume",
    "VLLM_CACHE_ROOT": "the container of the endpoint: the cache of vLLM and of torch.compile",
    # the identity of a Modal container, set by Modal and read by the watchdog around a memory snapshot (MFL045 c2,
    # plan r2 E.2 and E.7); MODAL_IDENTITY_TOKEN, an OIDC token of the same list, is never read for its value
    "MODAL_IDENTITY_TOKEN": "the container of a Modal app: an OIDC token of its runtime, taken out of the environment "
                            "of vLLM and of the download by watchdog.vllm_env (MFL049 c), never read for its value",
    "MODAL_TASK_ID": "the container of a Modal app: its task, which a restore of a snapshot changes",
    "MODAL_ENABLE_SNAP_RESTORE": "the container of a Modal app: 1 in a task of a memory snapshot",
    "MODAL_REGION": "the container of a Modal app: its region",
    "MODAL_CLOUD_PROVIDER": "the container of a Modal app: its cloud",
    "MODAL_IMAGE_ID": "the container of a Modal app: its image",
    # the thresholds of the watchdog in the container (`watchdog.config`, `MDF_<RULE>` read dynamically): internal
    # overrides of a diagnosis of the container, never set by the arm nor by the user on this machine
    **{name: "the container of the endpoint: a threshold of the watchdog (watchdog.RULES, watchdog.DEFAULTS), an "
       "internal override" for name in ("MDF_W1_S", "MDF_W2_S", "MDF_W3_S", "MDF_W4_S", "MDF_W5_S", "MDF_TICK_S",
                                        "MDF_REQ_MAX_S")},
}
# Named in the guides as the key of an endpoint `user` of a provider (`key_env` of servers.json), not by the code
DOCUMENTED_KEYS = ("MODAL_PROXY_BEARER",)


def _watchdog_names():
    sys.path.insert(0, str(ROOT / "scripts" / "servers"))
    import watchdog
    keys = set(watchdog.RULES) | {k for d in watchdog.DEFAULTS.values() for k in d}
    return sorted("MDF_" + k.upper() for k in keys)


def _key_names_of_endpoints():
    from markdown_for_llms.engine import servers
    return sorted({k for v in servers.KEY_ENV.values() for k in v})


def _constant(module, name):
    def names():
        import importlib
        if module.startswith("scripts."):
            sys.path.insert(0, str(ROOT / "scripts" / "servers"))
            value = getattr(importlib.import_module(module.rsplit(".", 1)[1]), name)
        else:
            value = getattr(importlib.import_module(module), name)
        return [value] if isinstance(value, str) else list(value)
    return names


SESSION, WORKER_LOG = _constant("markdown_for_llms.bench.serverspend", "SESSION_ENV"), \
    _constant("scripts.servers.worker_logs", "LOG_ENV")
# Every read whose name is not a literal: {module: {expression: the names it can read}} (S-NB3).
DYNAMIC = {
    "markdown_for_llms/bench/serverspend.py": {"environ.get(SESSION_ENV)": SESSION},
    # the keys of an endpoint (`key_env`), whose default names are those of servers.KEY_ENV
    "markdown_for_llms/engine/servers.py": {"environ.get(key)": _key_names_of_endpoints,
                                            "environ[key]": _key_names_of_endpoints},
    "scripts/servers/modal_tasks.py": {"self.environ.get(wl.LOG_ENV)": WORKER_LOG,
                                       "self.environ.get(serverspend.SESSION_ENV)": SESSION},
    "scripts/servers/runpod_endpoint.py": {"self.environ.get(wl.LOG_ENV)": WORKER_LOG,
                                           "self.environ.get(serverspend.SESSION_ENV)": SESSION},
    "scripts/servers/start_series.py": {"self.environ.get(wl.LOG_ENV)": WORKER_LOG,
                                        "self.environ.get(serverspend.SESSION_ENV)": SESSION},
    "scripts/servers/watchdog.py": {"env.get('MDF_' + key.upper())": _watchdog_names,
                                    "env['MDF_' + key.upper()]": _watchdog_names},
    # the keys of the rented GPUs that the worker takes out of its environment (MFL023 A4): those of
    # gpu.providers.KEYS and the key_env of the endpoints, whose default names are those of servers.KEY_ENV
    "markdown_for_llms/worker.py": {"environ.pop(n, None)": lambda: sorted(
        set(_constant("markdown_for_llms.gpu.providers", "KEYS")()) | set(_key_names_of_endpoints()))},
    # the same modules at their place in the package (MFL023 A1): scripts/servers holds links to them
    "markdown_for_llms/gpu/modal_tasks.py": {"self.environ.get(wl.LOG_ENV)": WORKER_LOG,
                                             "self.environ.get(serverspend.SESSION_ENV)": SESSION},
    "markdown_for_llms/gpu/runpod_endpoint.py": {"self.environ.get(wl.LOG_ENV)": WORKER_LOG,
                                                 "self.environ.get(serverspend.SESSION_ENV)": SESSION},
    "markdown_for_llms/gpu/start_series.py": {"self.environ.get(wl.LOG_ENV)": WORKER_LOG,
                                              "self.environ.get(serverspend.SESSION_ENV)": SESSION},
    "markdown_for_llms/gpu/watchdog.py": {"env.get('MDF_' + key.upper())": _watchdog_names,
                                          "env['MDF_' + key.upper()]": _watchdog_names},
}


def _receiver(node):
    while isinstance(node, ast.Attribute):
        if node.attr in ENVISH:
            return node.attr
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _literal(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _name_node(n):
    """The node that names the variable of a read of the environment, or None for anything else."""
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args:
        if n.func.attr == "getenv" or (n.func.attr in ("get", "pop") and _receiver(n.func.value) in ENVISH):
            return n.args[0]
    elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load) and _receiver(n.value) in ENVISH:
        return n.slice
    elif (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn))
          and _receiver(n.comparators[0]) in ENVISH):
        return n.left
    return None


def dynamic_reads(tree):
    """The reads of the environment whose name is not a literal, as their source (`ast.unparse`)."""
    return {ast.unparse(n) for n in ast.walk(tree) if (node := _name_node(n)) is not None and _literal(node) is None}


def names_read(tree):
    """The environment variables a module reads, by the rules of the module docstring."""
    out = set()
    for n in ast.walk(tree):
        found = []
        node = _name_node(n)
        if node is not None:
            found.append(_literal(node))
        elif (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id.endswith("_ENV")):
            values = [n.value]
            while values:
                v = values.pop()
                if isinstance(v, (ast.List, ast.Tuple, ast.Set)):
                    values.extend(v.elts)
                elif isinstance(v, ast.Dict):
                    values.extend(v.values)
                else:
                    found.append(_literal(v))
        out.update(x for x in found if x and NAME.match(x))
    return out


def modules():
    for top in SCOPE:
        for path in sorted((ROOT / top).rglob("*.py")):
            yield str(path.relative_to(ROOT)), ast.parse(path.read_text(encoding="utf-8"), str(path))


def code_names():
    """{name: [module, ...]} over the scope: the literal reads, then the names of the dynamic ones (DYNAMIC)."""
    found = {}
    for where, tree in modules():
        names = set(names_read(tree))
        for expr in dynamic_reads(tree):
            resolve = DYNAMIC.get(where, {}).get(expr)
            if resolve is not None:
                names.update(resolve())
        for name in names:
            found.setdefault(name, []).append(where)
    return found


def unlisted_dynamic_reads():
    """{module: [expression]} of the dynamic reads that DYNAMIC does not list."""
    out = {}
    for where, tree in modules():
        unknown = sorted(e for e in dynamic_reads(tree) if e not in DYNAMIC.get(where, {}))
        if unknown:
            out[where] = unknown
    return out


def settings_names():
    from markdown_for_llms import settings
    from markdown_for_llms.bench import settings as bench_settings
    return (["MDF_" + f.name.upper() for f in dataclasses.fields(settings.Settings)]
            + [bench_settings.PREFIX + f.name.upper() for f in dataclasses.fields(bench_settings.BenchSettings)])


def key_names():
    from markdown_for_llms.engine import datalab, servers
    sys.path.insert(0, str(ROOT / "scripts" / "servers"))
    import providers
    return sorted({datalab.KEY_NAME, *[k for v in servers.KEY_ENV.values() for k in v], *providers.KEYS,
                   *DOCUMENTED_KEYS})


def example_lines():
    """{name: (commented, value)} of `.env.example`."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return {m.group(2): (bool(m.group(1)), m.group(3)) for m in LINE.finditer(text)}


class TestEnvExample(unittest.TestCase):
    def test_every_variable_the_code_reads_is_in_the_example_or_excluded(self):
        lines = example_lines()
        missing = {name: where for name, where in code_names().items() if name not in lines and name not in EXCLUDED}
        missing.update({name: ["settings"] for name in settings_names() if name not in lines})
        missing.update({name: ["keys"] for name in key_names() if name not in lines})
        self.assertEqual(missing, {})

    def test_no_dynamic_read_outside_its_table(self):
        """S-NB3: a read of the environment whose name is not a literal passes only when DYNAMIC lists it with
        the names it can read; and DYNAMIC lists only reads that are still in the code."""
        self.assertEqual(unlisted_dynamic_reads(), {})
        present = {where: dynamic_reads(tree) for where, tree in modules()}
        stale = {where: sorted(e for e in table if e not in present.get(where, set())) for where, table in DYNAMIC.items()}
        self.assertEqual({k: v for k, v in stale.items() if v}, {})
        for where, table in DYNAMIC.items():
            for expr, resolve in table.items():
                with self.subTest(read=where + ": " + expr):
                    self.assertTrue(resolve(), "no name")

    def test_the_watchdog_thresholds_are_internal(self):
        self.assertEqual(_watchdog_names(), ["MDF_REQ_MAX_S", "MDF_TICK_S", "MDF_W1_S", "MDF_W2_S", "MDF_W3_S",
                                             "MDF_W4_S", "MDF_W5_S"])
        self.assertTrue(all(n in EXCLUDED for n in _watchdog_names()))

    def test_every_exclusion_is_read_by_the_code_and_not_in_the_example(self):
        read, lines = code_names(), example_lines()
        self.assertEqual(sorted(n for n in EXCLUDED if n not in read), [])
        self.assertEqual(sorted(n for n in EXCLUDED if n in lines), [])

    def test_the_keys_by_name_only_commented_and_without_a_value(self):
        lines = example_lines()
        for name in key_names():
            with self.subTest(key=name):
                commented, value = lines.get(name, (None, None))
                self.assertIs(commented, True)
                self.assertEqual(value.strip(), "")

    def test_the_bench_commands_that_take_a_settings_file(self):
        """C-DA-NB1 of the review of A: the commands `.env.example` names as reading MDF_BENCH_CACHE_DIR from the
        settings file are those of `mdf bench` with an `--env-file`."""
        import argparse
        from markdown_for_llms.bench import cli as bench_cli
        sub = next(a for a in bench_cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction))
        taking = sorted(name for name, p in sub.choices.items() if any(a.dest == "env_file" for a in p._actions))
        text = " ".join(line.lstrip("# ").strip() for line in
                        (ROOT / ".env.example").read_text(encoding="utf-8").splitlines())
        found = re.search(r"by the commands that take a settings file \(([^)]*)\)", text)
        self.assertIsNotNone(found)
        self.assertEqual(sorted(n.strip() for n in found.group(1).split(",")), taking)

    def test_the_scan_finds_the_forms_of_a_read(self):
        tree = ast.parse("import os\n"
                         "a = os.environ.get('MDF_A')\nb = os.environ['MDF_B']\nc = self.environ.get('MDF_C')\n"
                         "d = env.get('MDF_D', '')\ne = os.getenv('MDF_E')\nf = 'MDF_F' in os.environ\n"
                         "LOG_ENV = 'MDF_G'\nKEY_ENV = {'p': ['MDF_H', 'MDF_I']}\n"
                         "os.environ['MDF_STORE'] = '1'\nh = headers.get('X_NOT')\ng = env.get(name)\n"
                         "i = env.get('MDF_' + key)\n")
        self.assertEqual(names_read(tree), {"MDF_A", "MDF_B", "MDF_C", "MDF_D", "MDF_E", "MDF_F", "MDF_G", "MDF_H",
                                            "MDF_I"})
        self.assertEqual(dynamic_reads(tree), {"env.get(name)", "env.get('MDF_' + key)"})


if __name__ == "__main__":
    unittest.main()
