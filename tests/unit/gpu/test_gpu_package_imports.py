"""The GPU session code as a package (MFL023 A1-A2): imported from `markdown_for_llms.gpu`, it never loads the
script form of its siblings (`providers`, `watchdog`, `worker_logs` as top-level modules), also on the paths that
import inside a method (the levers, the Modal and RunPod followers); the product imports none of the files that run
at the provider or measure, and the API process imports of the package only its data tables (`presets` and the `codes` it raises)."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
GPU = ROOT / "markdown_for_llms" / "gpu"
SCRIPT_FORM = ("providers", "session", "watchdog", "worker_logs", "drivers", "snapshot", "runpod_endpoint",
               "modal_tasks", "start_series", "modal_app", "modal_snap", "modal_populate")
# Files that run at the provider (Modal apps, the snapshot of the container) or that measure and manage the
# endpoints from the command line: moved as files only, never imported by the product (MFL023 plan r2 §3.1).
NOT_FOR_THE_PRODUCT = ("modal_app", "modal_snap", "modal_populate", "snapshot", "drivers", "runpod_endpoint",
                       "modal_tasks", "start_series")

# Runs in a fresh isolated process with the checkout first on sys.path: no network (a fake runner for the CLI of
# Modal, MockTransport for RunPod), prints the script-form modules loaded after each step.
PROBE = r'''
import json, socket, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
for name in ("connect", "connect_ex", "sendto"):
    setattr(socket.socket, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))
FORMS = json.loads(sys.argv[2])
steps = {}
def mark(step):
    steps[step] = sorted(n for n in FORMS if n in sys.modules)
import httpx
from markdown_for_llms.engine import envfile
from markdown_for_llms.gpu import session
pv = session.pv
mark("import")
facts = {"providers": pv.__name__, "here": str(pv.HERE), "watchdog_bytes": pv.watchdog_bytes() ==
         (Path(sys.argv[1]) / "markdown_for_llms/gpu/watchdog.py").read_bytes()}
facts["levers"] = bool(pv.levers_text(pv.DEFAULT_LEVERS, "modal"))
facts["watchdog_module"] = "markdown_for_llms.gpu.watchdog" in sys.modules
mark("levers")
facts["base_env"] = "MDF_LEVERS" in pv.base_env("", False, pv.DEFAULT_LEVERS)
mark("base_env")
tmp = Path(tempfile.mkdtemp(prefix="mdf-gpu-imports-"))
runner = lambda argv, **kw: types.SimpleNamespace(returncode=0, stdout="[]", stderr="")
f = pv.Modal("mdf-surya", runner=runner, environ={}).log_follower(tmp / "modal.partial", 0.0)
facts["modal_follower"] = type(f).__module__ + "." + type(f).__name__
mark("modal_follower")
def handler(request):
    if request.url.path.endswith("/serverless"):
        return httpx.Response(200, json={"endpoints": [{"id": "ep1", "name": "mdf-surya"}], "pagination": {}})
    return httpx.Response(200, json={"id": "ep1", "name": "mdf-surya"})
g = pv.RunPod(envfile.Secret("synthetic-key-0000"), "mdf-surya", transport=httpx.MockTransport(handler)).log_follower(
    tmp / "runpod.partial", 0.0)
facts["runpod_follower"] = type(g).__module__ + "." + type(g).__name__
mark("runpod_follower")
print(json.dumps({"steps": steps, "facts": facts}))
'''


def run(code, *args):
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT), *args], capture_output=True, text=True,
                         timeout=120)
    if out.returncode:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout)


class PackageImportTests(unittest.TestCase):
    def test_the_package_form_never_loads_the_script_form(self):
        result = run(PROBE, json.dumps(SCRIPT_FORM))
        for step, leaked in result["steps"].items():
            with self.subTest(step=step):
                self.assertEqual(leaked, [])
        facts = result["facts"]
        self.assertEqual(facts["providers"], "markdown_for_llms.gpu.providers")
        self.assertEqual(Path(facts["here"]), GPU)
        self.assertTrue(facts["watchdog_bytes"])
        self.assertTrue(facts["levers"])
        self.assertTrue(facts["watchdog_module"])
        self.assertTrue(facts["base_env"])
        self.assertEqual(facts["modal_follower"], "markdown_for_llms.gpu.worker_logs.ModalFollower")
        self.assertEqual(facts["runpod_follower"], "markdown_for_llms.gpu.worker_logs.Follower")

    def test_the_scripts_are_links_to_the_package(self):
        for name in SCRIPT_FORM:
            path = GPU / (name + ".py")
            link = ROOT / "scripts" / "servers" / path.name
            with self.subTest(module=path.name):
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.resolve(), path.resolve())

    def test_the_product_imports_no_file_of_the_provider_or_of_the_measures(self):
        def imported(path):
            names = set()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    names.update((base + "." + a.name).strip(".") for a in node.names)
                    names.add(base)
            return names
        for path in sorted((ROOT / "markdown_for_llms").rglob("*.py")):
            if path.parent == GPU and path.stem in NOT_FOR_THE_PRODUCT:
                continue
            names = imported(path)
            for module in NOT_FOR_THE_PRODUCT:
                with self.subTest(module=str(path.relative_to(ROOT)), imports=module):
                    self.assertFalse({module, "markdown_for_llms.gpu." + module, "gpu." + module} & names)

    def test_the_api_process_imports_only_the_tables_of_the_package(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import json, markdown_for_llms.api; "
                "print(json.dumps(sorted(m for m in sys.modules if m.startswith('markdown_for_llms.gpu'))))")
        loaded = set(run(code))
        self.assertLessEqual(loaded, {"markdown_for_llms.gpu", "markdown_for_llms.gpu.presets", "markdown_for_llms.gpu.codes"})


if __name__ == "__main__":
    unittest.main()
