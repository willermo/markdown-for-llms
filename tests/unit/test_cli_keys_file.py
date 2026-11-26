"""The settings and the keys are two files in the containers of the product (MFL051, plan r2.1 §5.1, J.5.2).

- `--env-file` gives the settings (its `MDF_` lines); `--keys-file` the keys, read by name at every use. With
  `--env-file` alone that file is both, as before.
- The key of Datalab comes from the keys file, never from the environment of the process: no key is in the
  environment of a container.
- `mdf settings-env` is the `config` service: it validates the env file of the user and writes `settings.env`,
  the settings every service reads, with the paths of the containers. No key reaches that file, the names of
  `make` and compose (`MDF_COMPOSE_*`) stay out, and a path written in the env file is refused.
- There is no `MDF_ENV_FILE`: a name the product does not know is refused like any other.
- `mdf serve --uds PATH`: the socket of the API is of the user alone, 0600 in a directory 0700.
- The worker starts also with a keys file that is not safe (rilievo I): the routes with a key say so.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import unittest
from unittest import mock

from markdown_for_llms import cli
from markdown_for_llms.settings import ConfigurationError, Settings, container_settings

SECRET = "dl-synthetic-0123456789"
FIXED = {"MDF_DATA_ROOT": "/data", "MDF_MODEL_ROOT": "/models/marker", "MDF_SERVERS": "/state/config/servers.json",
         "MDF_GPU_LEDGER_DIR": "/state/gpu-ledger"}


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.pdf = Path(__file__).resolve().parents[1] / "fixtures" / "conversion-corpus" / "inputs" / "C06-pdf-text-it.pdf"

    def file(self, name, text, mode=0o600):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        path.chmod(mode)
        return path

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main(argv)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()


class KeysFileTests(Base):
    def convert(self, *files, environ=None):
        argv = [*files, "--data-root", str(self.root / "data"), "--model-root", str(self.root / "models"), "convert",
                str(self.pdf), "--route", "datalab", "--out", str(self.root / "out")]
        with mock.patch.dict(os.environ, environ or {}):
            code, out, _ = self.run_cli(argv)
        self.assertNotIn(SECRET, out)
        return code, json.loads(out)

    def test_the_key_of_datalab_comes_from_the_keys_file(self):
        settings = self.file("settings.env", "MDF_MAX_PAGES=10\n")
        keys = self.file("keys", "DATALAB_API_KEY=%s\n" % SECRET)
        seen = []

        def convert(*args, **kw):
            seen.append(kw["datalab_key"])
            raise cli.EngineError(cli.EngineError.__init__.__globals__["Code"].OUTPUT_INVALID, "stop here")
        with mock.patch("markdown_for_llms.engine.core.convert", convert):
            code, answer = self.convert("--env-file", str(settings), "--keys-file", str(keys))
        self.assertEqual((code, answer["status"]), (2, "refused"))
        self.assertEqual(seen[0].header(), {"X-API-Key": SECRET})
        self.assertEqual(repr(seen[0]), "Key(***)")

    def test_a_keys_file_that_is_not_safe_or_has_no_key(self):
        settings = self.file("settings.env", "MDF_MAX_PAGES=10\nDATALAB_API_KEY=%s\n" % SECRET)
        loose = self.file("loose", "DATALAB_API_KEY=%s\n" % SECRET, 0o644)
        empty = self.file("empty", "OTHER_API_KEY=x\n")
        # the file of the settings is not where the keys are looked for, once a keys file is named
        self.assertEqual(self.convert("--env-file", str(settings), "--keys-file", str(loose)),
                         (1, {"status": "refused", "error": "datalab_key_unsafe", "out": str(self.root / "out")}))
        self.assertEqual(self.convert("--env-file", str(settings), "--keys-file", str(empty))[1]["error"], "datalab_key_missing")
        self.assertEqual(self.convert("--keys-file", str(self.root / "absent"))[1]["error"], "datalab_key_unsafe")

    def test_the_environment_never_gives_the_key(self):
        environ = {"DATALAB_API_KEY": SECRET}
        self.assertEqual(self.convert(environ=environ)[1]["error"], "datalab_key_missing")
        empty = self.file("empty", "OTHER_API_KEY=x\n")
        self.assertEqual(self.convert("--keys-file", str(empty), environ=environ)[1]["error"], "datalab_key_missing")

    def test_with_the_env_file_alone_that_file_is_both(self):
        both = self.file("both.env", "MDF_MAX_PAGES=10\nDATALAB_API_KEY=%s\n" % SECRET)
        seen = []

        def convert(*args, **kw):
            seen.append(kw["datalab_key"])
            raise cli.EngineError(cli.EngineError.__init__.__globals__["Code"].OUTPUT_INVALID, "stop here")
        with mock.patch("markdown_for_llms.engine.core.convert", convert):
            self.convert("--env-file", str(both))
        self.assertEqual(seen[0].header(), {"X-API-Key": SECRET})

    def test_the_worker_starts_with_a_keys_file_that_is_not_safe(self):
        """Rilievo I: before MFL051 `mdf worker` left with 1 (`server_key_unsafe`). Its settings no longer come
        from that file: it starts, and the file is refused at every read of a key."""
        settings = self.file("settings.env", "MDF_MAX_PAGES=10\n")
        link = self.root / "link"
        link.symlink_to(self.file("target", "DATALAB_API_KEY=%s\n" % SECRET))
        for keys in (self.file("loose", "DATALAB_API_KEY=%s\n" % SECRET, 0o644), link, self.root / "absent"):
            with self.subTest(keys=keys.name), mock.patch("markdown_for_llms.worker.Worker") as worker:
                code, out, _ = self.run_cli(["--env-file", str(settings), "--keys-file", str(keys), "--data-root",
                                             str(self.root / "data"), "--model-root", str(self.root / "models"),
                                             "worker", "--once"])
                self.assertEqual((code, out), (0, ""))
                self.assertEqual(worker.call_args.kwargs, {"env_file": str(keys)})
                worker.return_value.run.assert_called_once_with(True)


class SettingsEnvTests(Base):
    USER = ("# my settings\nMDF_PORT=23467\nexport MDF_MAX_PAGES=20\nMDF_COMPOSE_DATA=/srv/archive\n"
            "MDF_BENCH_TOOL_MEMORY_MB=2048\nDATALAB_API_KEY=%s\nRUNPOD_API_KEY=rp-synthetic\nOTHER=1\n" % SECRET)

    def write(self, text, *extra):
        source = self.file("env", text)
        out = self.root / "settings" / "settings.env"
        out.parent.mkdir(exist_ok=True)
        # the archive of the containers is /data; here a folder of the test, which no archive of the host can be
        self.fixed = dict(FIXED, MDF_DATA_ROOT=str(self.root / "archive"))
        code, answer, _ = self.run_cli(["--env-file", str(source), "settings-env", "--out", str(out),
                                        *["--set=%s=%s" % item for item in self.fixed.items()], *extra])
        return code, json.loads(answer), out

    def test_the_settings_of_the_services_without_keys_and_without_the_names_of_compose(self):
        code, answer, out = self.write(self.USER)
        self.assertEqual(code, 0)
        text = out.read_text()
        self.assertEqual(text.splitlines(), ["MDF_PORT=23467", "export MDF_MAX_PAGES=20", "MDF_BENCH_TOOL_MEMORY_MB=2048",
                                             "MDF_DATA_ROOT=%s/archive" % self.root, "MDF_GPU_LEDGER_DIR=/state/gpu-ledger",
                                             "MDF_MODEL_ROOT=/models/marker", "MDF_SERVERS=/state/config/servers.json"])
        self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o600)
        self.assertEqual(os.listdir(out.parent), ["settings.env"])                      # nothing partial left
        for forbidden in (SECRET, "rp-synthetic", "API_KEY", "MDF_COMPOSE_"):
            self.assertNotIn(forbidden, text)
            self.assertNotIn(forbidden, json.dumps(answer))
        # what the services read from it
        settings = Settings.load(out, environ={})
        self.assertEqual((settings.port, settings.max_pages, settings.data_root, settings.model_root, settings.servers,
                          settings.gpu_ledger_dir),
                         (23467, 20, str(self.root / "archive"), "/models/marker", "/state/config/servers.json",
                          "/state/gpu-ledger"))

    def test_a_copy_of_the_example_is_accepted_as_it_is(self):
        """`cp .env.example .env`, then `make up`: the paths of the containers are commented in the example."""
        example = Path(__file__).resolve().parents[2] / ".env.example"
        code, answer, out = self.write(example.read_text(encoding="utf-8"))
        self.assertEqual(code, 0, answer)
        written = out.read_text()
        for name in ("MDF_PORT=23465", "MDF_MAX_PAGES=50", "MDF_DATALAB_CAP_USD=0"):
            self.assertIn(name, written)
        self.assertNotIn("MDF_COMPOSE_", written)

    def test_a_path_written_in_the_env_file_is_refused(self):
        for name in FIXED:
            with self.subTest(name=name):
                code, answer, out = self.write("MDF_PORT=23467\n%s=/somewhere/private\n" % name)
                self.assertEqual((code, answer["code"]), (1, "invalid_setting"))
                self.assertIn(name, answer["error"])
                self.assertNotIn("/somewhere/private", answer["error"])                 # the name, never the value
                self.assertFalse(out.exists())

    def test_what_the_services_would_refuse_is_refused_here_and_nothing_is_written(self):
        for text in ("MDF_PORT=not-a-port\n", "MDF_ENV_FILE=/run/mdf/env\n", "MDF_UNKNOWN=1\n", "MDF_HOST=0.0.0.0\n",
                     "MDF_DATALAB_CAP_USD=-1\n"):
            with self.subTest(text=text.strip()):
                code, answer, out = self.write(text)
                self.assertEqual((code, answer["code"]), (1, "invalid_setting"))
                self.assertNotIn(text.partition("=")[2].strip(), answer["error"])
                self.assertFalse(out.exists())

    def test_a_refused_file_leaves_the_settings_of_before(self):
        code, _, out = self.write("MDF_PORT=23467\n")
        self.assertEqual(code, 0)
        before = out.read_text()
        code, _, out = self.write("MDF_PORT=x\n")
        self.assertEqual((code, out.read_text()), (1, before))

    def test_usage(self):
        out = self.root / "settings.env"
        self.assertEqual(self.run_cli(["settings-env", "--out", str(out)])[0], 2)       # no env file to validate
        source = self.file("env", "MDF_PORT=23467\n")
        for bad in ("--set=MDF_DATA_ROOT", "--set=DATA_ROOT=/data", "--set=MDF_DATA_ROOT="):
            self.assertEqual(self.run_cli(["--env-file", str(source), "settings-env", "--out", str(out), bad])[0], 2)
        self.assertFalse(out.exists())
        with self.assertRaises(ConfigurationError):
            container_settings(source, {"MDF_NOT_A_SETTING": "1"})
        with self.assertRaises(ConfigurationError):
            container_settings(self.root / "absent", FIXED)

    def test_the_names_of_make_and_compose_are_not_settings_and_there_is_no_mdf_env_file(self):
        settings = Settings.load(environ={"MDF_COMPOSE_DATA": "/srv/archive", "MDF_COMPOSE_UID": "1000", "MDF_PORT": "23467"})
        self.assertEqual(settings.port, 23467)
        with self.assertRaises(ConfigurationError):
            Settings.load(environ={"MDF_ENV_FILE": "/run/mdf/env"})
        code, out, _ = self.run_cli(["--env-file", str(self.file("env", "MDF_ENV_FILE=/run/mdf/env\n")), "--data-root",
                                     str(self.root / "data"), "jobs"])
        self.assertEqual((code, json.loads(out)), (1, {"error": "Unknown MDF setting"}))


class ArchiveGateTests(Base):
    """A service whose settings its archive refuses leaves with 1, and compose would start it again for ever (the
    note of the supervision on A2-M2). The `config` step makes the comparison first, without writing anything:
    `make up` and `make restart` stop there, with the reason and the way out."""

    def setUp(self):
        super().setUp()
        from markdown_for_llms.storage import Storage
        self.data = self.root / "data"
        self.created = Settings(data_root=str(self.data), model_root=str(self.root / "models"), max_pages=50)
        Storage(self.created).close()                                           # an archive with the identity of 50 pages
        self.out = self.root / "settings" / "settings.env"
        self.out.parent.mkdir()
        self.fixed = dict(FIXED, MDF_DATA_ROOT=str(self.data))

    def config(self, text, *extra):
        source = self.file("env", text)
        code, answer, _ = self.run_cli(["--env-file", str(source), "settings-env", "--out", str(self.out),
                                        *["--set=%s=%s" % item for item in self.fixed.items()], *extra])
        return code, json.loads(answer)

    def test_the_settings_the_archive_was_created_with_pass(self):
        self.assertEqual(self.config("MDF_MAX_PAGES=50\n")[0], 0)
        self.assertIn("MDF_MAX_PAGES=50", self.out.read_text())
        # the settings that are not part of the identity change freely
        self.assertEqual(self.config("MDF_MAX_PAGES=50\nMDF_PORT=23467\nMDF_TIMEOUT_SECONDS=600\nMDF_DATALAB_CAP_USD=0.05\n")[0], 0)
        self.assertIn("MDF_PORT=23467", self.out.read_text())

    def test_another_identity_is_refused_before_any_service_with_what_changed_and_the_way_out(self):
        self.assertEqual(self.config("MDF_MAX_PAGES=50\nMDF_THREADS=4\n")[0], 0)        # the last start that passed
        before = self.out.read_text()
        code, answer = self.config("MDF_MAX_PAGES=40\nMDF_THREADS=2\nMDF_PORT=23467\n")
        self.assertEqual((code, answer["code"]), (1, "invalid_setting"))
        self.assertIn("changed since the last start: MDF_MAX_PAGES, MDF_THREADS)", answer["error"])   # names, not MDF_PORT
        for way_out in ("Put the value of before back", "MDF_COMPOSE_DATA", "mdf configure", "Nothing was started or changed"):
            self.assertIn(way_out, answer["error"])
        # the change of the quota: its steps in their order, and where they are written (SOL-A2-B2)
        steps = [answer["error"].index(step) for step in (
            "`make down`", "`mdf configure --expected-config ID --storage-mib N`", '"config" in identity.json',
            "the same N in the env file", "`make up`", 'docs/how-to/run-in-containers.md, "Settings that cannot change"')]
        self.assertEqual(steps, sorted(steps))
        self.assertNotIn("40", answer["error"])                                 # never a value
        self.assertEqual(self.out.read_text(), before)                          # the services keep the settings of before
        self.assertEqual(os.listdir(self.out.parent), ["settings.env"])
        self.assertEqual(sorted(p.name for p in self.data.iterdir()), sorted(p.name for p in self.data.iterdir()))

    def listing(self):
        return sorted((str(p.relative_to(self.data)), p.stat().st_mtime_ns, p.stat().st_size) for p in self.data.rglob("*"))

    def test_the_quota_changes_with_the_steps_the_refusal_names(self):
        """SOL-A2-B2, the procedure of the guide: the product stopped, `mdf configure` with the settings of the last
        start (`settings.env`), the same quota in the env file, `make up`."""
        env = "MDF_MAX_PAGES=50\n"
        self.assertEqual(self.config(env)[0], 0)                                # the last start: settings.env
        identity = json.loads((self.data / "identity.json").read_text())["config"]
        code, answer = self.config(env + "MDF_STORAGE_MIB=30000\n")             # the quota changed in the env file alone
        self.assertEqual(code, 1)
        self.assertIn("(changed since the last start: MDF_STORAGE_MIB)", answer["error"])
        self.assertNotIn("MDF_STORAGE_MIB=30000", self.out.read_text())
        code, out, _ = self.run_cli(["--env-file", str(self.out), "configure", "--expected-config", identity,
                                     "--storage-mib", "30000"])
        self.assertEqual(code, 0, out)
        changed = json.loads(out)
        self.assertEqual((changed["previous_config"], changed["settings"]["storage_mib"]), (identity, 30000))
        stored = json.loads((self.data / "identity.json").read_text())
        self.assertEqual(stored["config"], changed["config"])
        self.assertNotEqual(stored["config"], identity)
        # one change at a time (the limit the guide declares): until the next start the settings of the last one
        # are those of the quota of before, and the archive refuses them
        code, out, _ = self.run_cli(["--env-file", str(self.out), "configure", "--expected-config", stored["config"],
                                     "--storage-mib", "40000"])
        self.assertEqual(code, 1)
        self.assertIn("Storage configuration mismatch", json.loads(out)["error"])
        self.assertEqual(json.loads((self.data / "identity.json").read_text()), stored)
        # `make up`: the same quota in the env file passes, another one or the one of before does not
        self.assertEqual(self.config(env + "MDF_STORAGE_MIB=30000\n")[0], 0)
        self.assertIn("MDF_STORAGE_MIB=30000", self.out.read_text())
        self.assertEqual(self.config(env + "MDF_STORAGE_MIB=40000\n")[0], 1)
        self.assertEqual(self.config(env)[0], 1)

    def test_without_the_settings_of_the_last_start_the_two_commands_of_the_archive_say_what_to_do(self):
        """`make clean` removes `settings.env` with its volume: the commands that open the archive with the settings
        of the last start refuse, closed, before they open anything."""
        before = self.listing()
        for command in (["configure", "--expected-config", "0f" * 32, "--storage-mib", "30000"],
                        ["downgrade-storage", "--expected-config", "0f" * 32]):
            with self.subTest(command=command[0]):
                code, out, _ = self.run_cli(["--env-file", str(self.out), *command])
                answer = json.loads(out)
                self.assertEqual((code, answer["code"]), (1, "invalid_setting"))
                self.assertIn("mdf " + command[0], answer["error"])
                steps = [answer["error"].index(step) for step in ("put the env file back", "`make up`", "`make down`",
                                                                  "run this command again")]
                self.assertEqual(steps, sorted(steps))
                self.assertIn("Nothing was opened or changed", answer["error"])
                self.assertNotIn(str(self.out), answer["error"])                # a path of the machine is not said
        self.assertEqual(self.listing(), before)
        self.assertFalse(self.out.exists())

    def test_without_a_previous_start_the_refusal_names_no_setting_and_writes_nothing(self):
        code, answer = self.config("MDF_MAX_PAGES=40\n")
        self.assertEqual(code, 1)
        self.assertIn("refuses these", answer["error"])
        self.assertNotIn("changed since", answer["error"])
        self.assertFalse(self.out.exists())

    def test_the_check_of_make_status_writes_nothing(self):
        self.assertEqual(self.config("MDF_MAX_PAGES=50\n", "--check"), (0, {"settings": str(self.out), "checked": True}))
        self.assertFalse(self.out.exists())
        self.assertEqual(self.config("MDF_MAX_PAGES=50\n", "--write")[0], 0)
        before = self.out.read_text()
        code, answer = self.config("MDF_MAX_PAGES=40\n", "--check")
        self.assertEqual((code, self.out.read_text()), (1, before))
        self.assertIn("MDF_MAX_PAGES", answer["error"])
        self.assertEqual(self.config("MDF_MAX_PAGES=50\nMDF_PORT=23467\n", "--check")[0], 0)
        self.assertEqual(self.out.read_text(), before)                          # a valid change too is not written by a check

    def test_the_archive_is_only_read(self):
        before = self.listing()
        self.config("MDF_MAX_PAGES=50\n")
        self.config("MDF_MAX_PAGES=40\n")
        self.assertEqual(self.listing(), before)

    def test_a_new_archive_and_a_folder_that_is_not_private(self):
        fresh = self.root / "fresh"
        self.fixed["MDF_DATA_ROOT"] = str(fresh)
        self.assertEqual(self.config("MDF_MAX_PAGES=40\n")[0], 0)                # no folder yet
        fresh.mkdir(mode=0o700)
        self.assertEqual(self.config("MDF_MAX_PAGES=40\n")[0], 0)                # an empty folder: the first service creates the archive
        fresh.chmod(0o775)
        code, answer = self.config("MDF_MAX_PAGES=40\n")
        self.assertEqual(code, 1)
        self.assertIn("not writable by others", answer["error"])
        fresh.chmod(0o700)
        (fresh / "identity.json").write_text("{not json")
        self.assertIn("cannot be read", self.config("MDF_MAX_PAGES=40\n")[1]["error"])

    def test_the_worker_and_the_api_would_have_been_refused_by_the_same_archive(self):
        """What the gate prevents: the same settings, given to a service, end in the error of the archive."""
        from markdown_for_llms.storage import Storage, StorageError
        with self.assertRaises(StorageError) as ctx:
            Storage(Settings(data_root=str(self.data), model_root=str(self.root / "models"), max_pages=40))
        self.assertIn("Storage configuration mismatch", str(ctx.exception))


class SocketTests(Base):
    def test_the_socket_of_the_api_is_of_the_user_alone(self):
        private = self.root / "api"
        private.mkdir(mode=0o700)
        path = private / "api.sock"
        sock = cli.private_socket(path)
        self.addCleanup(sock.close)
        info = os.lstat(path)
        self.assertTrue(stat.S_ISSOCK(info.st_mode))
        self.assertEqual((stat.S_IMODE(info.st_mode), info.st_uid), (0o600, os.getuid()))
        self.assertEqual(sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN), 1)       # it listens
        again = cli.private_socket(path)                                        # the socket of a previous process
        again.close()
        self.assertEqual(stat.S_IMODE(os.lstat(path).st_mode), 0o600)

    def test_a_directory_others_can_enter_or_something_else_at_the_path_is_refused(self):
        shared = self.root / "shared"
        shared.mkdir(mode=0o755)
        with self.assertRaises(ValueError):
            cli.private_socket(shared / "api.sock")
        private = self.root / "api"
        private.mkdir(mode=0o700)
        (private / "api.sock").write_text("not a socket")
        with self.assertRaises(ValueError):
            cli.private_socket(private / "api.sock")
        self.assertEqual((private / "api.sock").read_text(), "not a socket")
        with self.assertRaises(OSError):
            cli.private_socket(self.root / "absent" / "api.sock")

    def test_serve_on_a_socket_gives_uvicorn_the_descriptor_and_no_port(self):
        private = self.root / "api"
        private.mkdir(mode=0o700)
        with mock.patch("uvicorn.run") as run, mock.patch("markdown_for_llms.api.create_app", return_value="app"):
            code, _, _ = self.run_cli(["--data-root", str(self.root / "data"), "--model-root", str(self.root / "models"),
                                       "serve", "--uds", str(private / "api.sock")])
        self.assertEqual(code, 0)
        args, kwargs = run.call_args
        self.assertEqual((args, set(kwargs)), (("app",), {"fd"}))
        self.assertEqual(stat.S_IMODE(os.lstat(private / "api.sock").st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
