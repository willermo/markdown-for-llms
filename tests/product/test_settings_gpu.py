"""The settings of the rented GPUs (MFL023 A4, C-B3 and C-NB2 of the plan r1): operational, so the identity of the
archive does not change; the presets an instance may enable; absolute paths that never overlap the archive and never
appear in `public()`; the ledger outside the archive; the settings that never read a key from the env file; the env file
of the worker checked before anything reads it, and the keys of the GPUs taken out of its environment."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from markdown_for_llms import settings as settings_module
from markdown_for_llms.capacity import archive_sizes
from markdown_for_llms.cli import main
from markdown_for_llms.engine import servers
from markdown_for_llms.services import Service
from markdown_for_llms.settings import ConfigurationError, Settings
from markdown_for_llms.worker import Worker, take_gpu_keys

# The fields of the identity of schema 3 at fe6b952, frozen here independently of the code under test.
IDENTITY_FIELDS = ("data_root", "host", "memory_mib", "upload_mib", "max_pages", "output_mib", "log_mib", "storage_mib",
                   "reserve_mib", "max_pixels", "max_dimension", "threads", "poll_seconds", "lease_seconds",
                   "grace_seconds", "process_limit", "bind_profile", "review_provider")
GPU_VALUES = {"gpu_presets": "modal-base, runpod-serverless-volume-dc", "gpu_cap_usd_modal": "3",
              "gpu_cap_usd_runpod": "2", "gpu_session_cap_usd": "0.50"}


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.base = dict(data_root=str(self.tmp / "data"), model_root=str(self.tmp / "models"),
                         gpu_ledger_dir=str(self.tmp / "ledger"), servers=str(self.tmp / "servers.json"))

    def refused(self, code=None, **values):
        with self.assertRaises(ConfigurationError) as ctx:
            Settings(**dict(self.base, **values))
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception


class SettingsTests(Base):
    def test_defaults(self):
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.tmp / "cfg"), "XDG_DATA_HOME": str(self.tmp / "share")}):
            s = Settings(data_root=str(self.tmp / "data"), model_root=str(self.tmp / "models"))
            self.assertEqual(s.servers, str(servers.default_path()))
        self.assertEqual(s.gpu_ledger_dir, str(self.tmp / "share" / "markdown-for-llms" / "gpu-ledger"))
        self.assertEqual((s.gpu_presets, s.gpu_cap_usd_modal, s.gpu_cap_usd_runpod, s.gpu_session_cap_usd),
                         ("", "0", "0", "0"))
        self.assertEqual(s.gpu_presets_enabled(), [])

    def test_the_identity_does_not_change(self):
        plain = Settings(**self.base)
        gpu = Settings(**dict(self.base, **GPU_VALUES, gpu_ledger_dir=str(self.tmp / "other"),
                              servers=str(self.tmp / "other.json")))
        self.assertEqual(plain.identity(), gpu.identity())
        frozen = {k: getattr(plain, k) for k in IDENTITY_FIELDS}
        frozen["identity_schema"] = 3
        self.assertEqual(plain.identity(), hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest())
        self.assertTrue(settings_module.GPU <= settings_module.OPERATIONAL)

    def test_the_presets_enabled(self):
        s = Settings(**dict(self.base, **GPU_VALUES))
        self.assertEqual(s.gpu_presets, "modal-base,runpod-serverless-volume-dc")
        self.assertEqual(s.gpu_presets_enabled(), ["modal-base", "runpod-serverless-volume-dc"])
        for value, code in (("runpod-serverless-volume-eu-se-1", "gpu_preset_superseded"),
                            ("secret-preset-name", "gpu_preset_unknown"), ("modal-base,modal-base", "gpu_preset_invalid"),
                            ("x" * 2000, "gpu_preset_invalid")):
            with self.subTest(value=value[:40]):
                err = self.refused(code, gpu_presets=value)
                self.assertNotIn("secret-preset-name", str(err))

    def test_amounts(self):
        for name in ("gpu_cap_usd_modal", "gpu_cap_usd_runpod", "gpu_session_cap_usd"):
            for value in ("-1", "abc", "NaN", "Infinity", 3, "1" * 40):
                with self.subTest(name=name, value=value):
                    self.refused(**{name: value})

    def test_paths_and_overlaps(self):
        s = Settings(**dict(self.base, gpu_ledger_dir="relative/ledger", servers="relative/servers.json"))
        self.assertTrue(Path(s.gpu_ledger_dir).is_absolute() and Path(s.servers).is_absolute())
        self.assertIsInstance(s.servers, str)
        data, models = self.base["data_root"], self.base["model_root"]
        for values in ({"gpu_ledger_dir": data + "/ledger"}, {"gpu_ledger_dir": data},
                       {"gpu_ledger_dir": str(self.tmp)}, {"gpu_ledger_dir": models + "/ledger"},
                       {"servers": data + "/servers.json"}, {"gpu_ledger_dir": ""}, {"servers": "a\0b"}):
            with self.subTest(values=values):
                self.refused(**values)

    def test_public_holds_no_path(self):
        public = Settings(**dict(self.base, **GPU_VALUES)).public()
        for name in ("data_root", "model_root", "servers", "gpu_ledger_dir"):
            self.assertNotIn(name, public)
        self.assertEqual(public["gpu_presets"], "modal-base,runpod-serverless-volume-dc")
        self.assertNotIn(str(self.tmp), json.dumps(public))

    def test_the_ledger_is_outside_the_archive(self):
        s = Settings(**dict(self.base, upload_mib=1, output_mib=3, log_mib=1, storage_mib=16, reserve_mib=1))
        service = Service(s)
        before = archive_sizes(service.storage)
        Path(s.gpu_ledger_dir).mkdir()
        (Path(s.gpu_ledger_dir) / "spend.jsonl").write_bytes(b"x" * 1048576)
        self.assertEqual(archive_sizes(service.storage), before)
        service.storage.close()
        shutil.rmtree(s.data_root)                                    # the archive recreated: the ledger stays
        Service(s).storage.close()
        self.assertEqual((Path(s.gpu_ledger_dir) / "spend.jsonl").stat().st_size, 1048576)


class EnvFileTests(Base):
    def env(self, text, mode=0o600):
        path = self.tmp / "mdf.env"
        path.write_text(text)
        path.chmod(mode)
        return path

    def test_the_settings_never_read_a_key(self):
        path = self.env("# keys\nRUNPOD_API_KEY=rp-synthetic-0000\nexport MODAL_PROXY_TOKEN_SECRET='ws-synthetic'\n"
                        "MDF_GPU_CAP_USD_RUNPOD=2\nexport MDF_GPU_PRESETS=runpod-serverless-volume-dc\n")
        seen = []
        import dotenv
        real = dotenv.dotenv_values

        def spy(*args, **kwargs):
            stream = kwargs.get("stream")
            seen.append(stream.getvalue() if stream is not None else Path(args[0]).read_text())
            return real(*args, **kwargs)
        with patch.object(dotenv, "dotenv_values", side_effect=spy):
            s = Settings.load(path, {"data_root": self.base["data_root"], "model_root": self.base["model_root"],
                                     "gpu_ledger_dir": self.base["gpu_ledger_dir"], "servers": self.base["servers"]},
                              environ={})
        self.assertEqual((s.gpu_cap_usd_runpod, s.gpu_presets), ("2", "runpod-serverless-volume-dc"))
        self.assertEqual(len(seen), 1)
        self.assertNotIn("synthetic", seen[0])
        self.assertNotIn("RUNPOD_API_KEY", seen[0])

    def run_worker(self, path):
        calls = []

        class FakeWorker:
            def __init__(self, settings, env_file=None):
                calls.append(env_file)
                self.stop = False

            def run(self, once):
                calls.append(("run", once))
        out = io.StringIO()
        with patch("markdown_for_llms.worker.Worker", FakeWorker), redirect_stdout(out):
            code = main(["--env-file", str(path), "--data-root", self.base["data_root"], "--model-root",
                         self.base["model_root"], "worker", "--once"])
        return code, out.getvalue(), calls

    def test_the_env_file_of_the_worker_is_checked_first(self):
        good = self.env("MDF_POLL_SECONDS=1\n")
        code, _, calls = self.run_worker(good)
        self.assertEqual((code, calls), (0, [str(good.absolute()), ("run", True)]))
        link = self.tmp / "link.env"
        link.symlink_to(good)
        fifo = self.tmp / "fifo.env"
        os.mkfifo(fifo)
        for path, mode in ((good, 0o644), (link, 0o600), (fifo, None)):
            with self.subTest(path=path.name, mode=mode):
                if mode is not None:
                    good.chmod(mode)
                with patch.object(Settings, "load", side_effect=AssertionError("read before the check")):
                    code, out, calls = self.run_worker(path)
                self.assertEqual((code, json.loads(out)["code"], calls), (1, "server_key_unsafe", []))
                good.chmod(0o600)

    def test_the_keys_of_the_gpus_leave_the_environment_of_the_worker(self):
        conf = self.tmp / "servers.json"
        conf.write_text(json.dumps({"default": "rp", "servers": {
            "rp": {"kind": "runpod", "preset": "runpod-serverless-volume-dc", "resource": "mdf-surya",
                   "url": "https://ep.api.runpod.ai", "key_env": ["TEAM_RUNPOD_API_KEY"]},
            "mine": {"kind": "user", "url": "https://gpu.example.org", "key_env": "MY_GPU_KEY"}}}))
        s = Settings(**dict(self.base, servers=str(conf)))
        environ = {"RUNPOD_API_KEY": "a", "MODAL_PROXY_TOKEN_ID": "b", "MODAL_PROXY_TOKEN_SECRET": "c",
                   "MODAL_PROXY_BEARER": "d", "TEAM_RUNPOD_API_KEY": "e", "MY_GPU_KEY": "f", "HOME": "/home/x"}
        taken = take_gpu_keys(s, environ)
        self.assertEqual(environ, {"HOME": "/home/x"})
        self.assertEqual(taken, sorted(["RUNPOD_API_KEY", "MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET",
                                        "MODAL_PROXY_BEARER", "TEAM_RUNPOD_API_KEY", "MY_GPU_KEY"]))
        conf.write_text("{not json")
        environ = {"RUNPOD_API_KEY": "a", "TEAM_RUNPOD_API_KEY": "e"}
        self.assertEqual(take_gpu_keys(s, environ), ["RUNPOD_API_KEY"])     # an unreadable file: the known names
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "rp-synthetic", "DATALAB_API_KEY": "dl-synthetic"}):
            worker = Worker(Settings(**dict(self.base, upload_mib=1, output_mib=3, log_mib=1, storage_mib=16,
                                            reserve_mib=1)), env_file=self.tmp / "keys.env")
            self.assertNotIn("RUNPOD_API_KEY", os.environ)
            self.assertNotIn("DATALAB_API_KEY", os.environ)
            self.assertEqual(worker.env_file, str(self.tmp / "keys.env"))
            self.assertIsNotNone(worker.datalab_key)
            worker.storage.close()

    def test_a_malformed_servers_file_never_stops_the_worker(self):
        """C-DA-NB3 of the review of A: a `servers` that is not an object, an entry or a `key_env` of another type are
        skipped, and only names of keys leave the environment (the rule of `engine.servers`: one of
        `envfile.KEY_SUFFIXES`, never an `MDF_` setting); the names of `gpu.providers.KEYS` always do."""
        conf = self.tmp / "servers.json"
        s = Settings(**dict(self.base, servers=str(conf)))
        docs = ({"servers": ["x"]}, {"servers": "x"}, {"servers": {"a": ["x"], "b": 7}},
                {"servers": {"a": {"kind": "user", "key_env": 123}}},
                {"servers": {"a": {"kind": "user", "key_env": True}}},
                {"servers": {"a": {"kind": "user", "key_env": "HOME"}}},
                {"servers": {"a": {"kind": "user", "key_env": ["PATH", 5, "MDF_DATA_ROOT", "MDF_X_KEY"]}}})
        for doc in docs:
            with self.subTest(doc=doc):
                conf.write_text(json.dumps(doc))
                environ = {"RUNPOD_API_KEY": "a", "HOME": "/home/x", "PATH": "/usr/bin", "MDF_DATA_ROOT": "/d",
                           "MDF_X_KEY": "m"}
                self.assertEqual(take_gpu_keys(s, environ), ["RUNPOD_API_KEY"])
                self.assertEqual(environ, {"HOME": "/home/x", "PATH": "/usr/bin", "MDF_DATA_ROOT": "/d",
                                           "MDF_X_KEY": "m"})
        conf.write_text(json.dumps({"servers": {"a": {"kind": "user", "key_env": ["HOME", "MY_GPU_KEY"]}}}))
        environ = {"HOME": "/home/x", "MY_GPU_KEY": "k"}
        self.assertEqual(take_gpu_keys(s, environ), ["MY_GPU_KEY"])
        self.assertEqual(environ, {"HOME": "/home/x"})


class HomePathTests(Base):
    def test_a_tilde_is_the_home_of_the_user(self):
        """SOL-A-N2 = C-DA-NB4 of the review of A: `~` at the start of MDF_SERVERS, MDF_GPU_LEDGER_DIR and MDF_MODEL_ROOT
        (the examples of `.env.example`) is the home of the user, as in `bench/settings.py`; the identity does not
        change, and MDF_DATA_ROOT, which is part of it, keeps its literal value."""
        env = self.tmp / "mdf.env"
        plain = "MDF_DATA_ROOT=%s\n" % (self.tmp / "data")
        env.write_text(plain + "MDF_SERVERS=~/s.json\nMDF_GPU_LEDGER_DIR=~/ledger\nMDF_MODEL_ROOT=~/models\n")
        with patch.dict(os.environ, {"HOME": str(self.tmp / "home")}):
            s = Settings.load(env, environ={})
            home = Path.home()
        self.assertEqual(home, self.tmp / "home")
        self.assertEqual((s.servers, s.gpu_ledger_dir, s.model_root),
                         (str(home / "s.json"), str(home / "ledger"), str(home / "models")))
        for value in (s.servers, s.gpu_ledger_dir, s.model_root):
            self.assertNotIn("~", Path(value).parts)
        env.write_text(plain)
        self.assertEqual(s.identity(), Settings.load(env, environ={}).identity())
        literal = Settings(**dict(self.base, data_root="~mdf-data"))
        self.assertEqual(literal.data_root, str(Path("~mdf-data").absolute()))


if __name__ == "__main__":
    unittest.main()
