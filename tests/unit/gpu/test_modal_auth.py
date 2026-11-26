"""The token of the Modal SDK in the containers (MFL051, plan r2.1 §7.1, J.7.1; D-2; rilievo E).

No container has a `~/.modal.toml`. `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` are two names of the keys file, and
each process of the Modal CLI gets them through a file of its own (`providers.modal_auth`):

- a unique file, mode 0600, in a directory of the user with mode 0700; the directory is a parameter, and its
  default is the tmpfs of the worker in the containers (`/run/mdf/secrets`);
- `MODAL_CONFIG_PATH` only in the environment of that child; no value in an environment, an argument or an error;
- a synchronous call and two streams at the same time have three different files;
- the file is removed after a normal exit, a timeout and an error; the files of a pid that no longer exists are
  removed at every use and at the start of the worker;
- without the token in the keys file: refused before the CLI starts, and never a file of the home;
- the log follower of a session and every call of `providers.Modal` pass the context.

Only processes that are not there: no CLI of Modal is started and nothing is imported from it.
"""
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from markdown_for_llms.gpu import lifecycle, providers, worker_logs

TOKEN_ID, TOKEN_SECRET = "ak-synthetic0123456789", "as-synthetic9876543210"
SECRETS = (TOKEN_ID, TOKEN_SECRET)


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.secrets = self.root / "secrets"
        self.secrets.mkdir(mode=0o700)
        self.keys = self.keys_file("MODAL_TOKEN_ID=%s\nMODAL_TOKEN_SECRET=%s\nMODAL_PROXY_TOKEN_ID=wk-x\n" % SECRETS)
        self.seen = []

    def keys_file(self, text, mode=0o600, name="env"):
        path = self.root / name
        path.write_text(text)
        path.chmod(mode)
        return str(path)

    def files(self):
        return sorted(os.listdir(self.secrets))

    def look(self, argv, env):
        """What a process of the CLI would see when it starts: its arguments, its environment, its file."""
        path = env.get("MODAL_CONFIG_PATH")
        record = {"argv": list(argv), "env": dict(env), "path": path}
        if path:
            info = os.stat(path)
            record.update(mode=stat.S_IMODE(info.st_mode), uid=info.st_uid, text=Path(path).read_text(), exists=True)
        self.seen.append(record)
        return record

    def runner(self, outcome=0):
        def run(argv, **kw):
            self.look(argv, kw["env"])
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
            if isinstance(outcome, BaseException):
                raise outcome
            return subprocess.CompletedProcess(argv, outcome, "[]", "")
        return run

    def assert_no_secret(self, *texts):
        for text in texts:
            for secret in SECRETS:
                self.assertNotIn(secret, str(text))


class ContextTests(Base):
    def test_a_unique_private_file_named_only_in_the_environment_of_the_child(self):
        before = dict(os.environ)
        with providers.modal_auth(self.keys, str(self.secrets)) as extra:
            self.assertEqual(list(extra), ["MODAL_CONFIG_PATH"])
            path = Path(extra["MODAL_CONFIG_PATH"])
            self.assertEqual(path.parent, self.secrets)
            self.assertRegex(path.name, r"^modal-%d-[A-Za-z0-9_]+\.toml$" % os.getpid())       # the pid of this process
            info = path.stat()
            self.assertEqual((stat.S_IMODE(info.st_mode), info.st_uid), (0o600, os.getuid()))
            self.assertEqual(path.read_text(), '[default]\ntoken_id = "%s"\ntoken_secret = "%s"\n' % SECRETS)
            self.assertEqual(dict(os.environ), before)                  # nothing in the environment of this process
            self.assert_no_secret(extra)
        self.assertEqual(self.files(), [])                              # removed when the block ends

    def test_the_default_directory_is_the_tmpfs_of_the_containers_and_it_is_a_parameter(self):
        self.assertEqual(providers.MODAL_AUTH_DIR, "/run/mdf/secrets")
        import inspect
        self.assertEqual(inspect.signature(providers.modal_auth).parameters["directory"].default, "/run/mdf/secrets")
        other = self.root / "elsewhere"
        other.mkdir(mode=0o700)
        with providers.modal_auth(self.keys, str(other)) as extra:
            self.assertEqual(Path(extra["MODAL_CONFIG_PATH"]).parent, other)
        self.assertEqual(self.files(), [])

    def test_a_directory_that_is_not_private_is_refused_and_nothing_is_written(self):
        shared = self.root / "shared"
        shared.mkdir(mode=0o755)
        link = self.root / "link"
        link.symlink_to(self.secrets)
        for directory in (shared, link, self.root / "absent", Path(self.keys)):
            with self.subTest(directory=directory.name), self.assertRaises(providers.ProviderError) as ctx:
                with providers.modal_auth(self.keys, str(directory)):
                    self.fail("entered")
            self.assert_no_secret(ctx.exception)
        self.assertEqual((self.files(), os.listdir(shared)), ([], []))

    def test_without_the_token_nothing_starts_and_no_file_of_the_home_is_looked_for(self):
        with self.assertRaises(providers.ProviderError):
            with providers.modal_auth(None, str(self.secrets)):
                self.fail("entered")
        cases = {"one name only": self.keys_file("MODAL_TOKEN_ID=%s\n" % TOKEN_ID, name="one"),
                 "no name": self.keys_file("MODAL_PROXY_TOKEN_ID=wk-x\n", name="none"),
                 "not private": self.keys_file("MODAL_TOKEN_ID=%s\nMODAL_TOKEN_SECRET=%s\n" % SECRETS, 0o644, "loose"),
                 "absent": str(self.root / "absent.env"),
                 "a value that would break the file": self.keys_file('MODAL_TOKEN_ID=a"b\nMODAL_TOKEN_SECRET=%s\n' % TOKEN_SECRET,
                                                                     name="odd")}
        home = self.root / "home"
        home.mkdir()
        (home / ".modal.toml").write_text('[default]\ntoken_id = "from-the-home"\n')
        for name, keys in cases.items():
            with self.subTest(case=name), mock.patch.dict(os.environ, {"HOME": str(home)}):
                with self.assertRaises(providers.ProviderError) as ctx:
                    providers.run_modal(["app", "list"], env={}, runner=self.runner(), auth=keys, auth_dir=str(self.secrets))
                self.assert_no_secret(ctx.exception)
                self.assertNotIn('a"b', str(ctx.exception))
        self.assertEqual((self.seen, self.files()), ([], []))            # no process was started

    def test_the_real_cli_never_starts_without_a_keys_file(self):
        with mock.patch.object(providers, "MODAL_SDK", ["/bin/true"]), self.assertRaises(providers.ProviderError) as ctx:
            providers.run_modal(["app", "list"], env={})                # the real runner, no `auth`
        self.assertIn("MODAL_TOKEN_ID", str(ctx.exception))
        with self.assertRaises(providers.ProviderError) as ctx:        # and it is the CLI of the image, not `uv`
            providers.run_modal(["app", "list"], env={}, auth=self.keys, auth_dir=str(self.secrets))
        self.assertEqual(providers.MODAL_SDK, ["/opt/modal/bin/modal"])
        for name in providers.MODAL_TOKEN:
            self.assertIn(name, providers.KEYS)                         # taken out of the environment of the worker

    def test_the_files_of_a_process_that_no_longer_exists_are_removed(self):
        dead = subprocess.Popen(["/bin/true"])
        dead.wait()
        stale = self.secrets / ("modal-%d-abc123.toml" % dead.pid)
        mine = self.secrets / ("modal-%d-zzz999.toml" % os.getpid())
        other = self.secrets / "notes.txt"
        for path in (stale, mine, other):
            path.write_text("x")
        self.assertEqual(providers.modal_auth_sweep(str(self.secrets)), [stale.name])
        self.assertEqual(self.files(), sorted([mine.name, other.name]))  # a live process and other files stay
        stale.write_text("x")
        with providers.modal_auth(self.keys, str(self.secrets)):        # also at every use
            self.assertFalse(stale.exists())
        self.assertEqual(providers.modal_auth_sweep(str(self.root / "absent")), [])


class SynchronousTests(Base):
    def test_the_file_is_there_for_the_process_and_gone_after_it(self):
        code, out, _ = providers.run_modal(["app", "list"], env={"PATH": "/usr/bin"}, runner=self.runner(),
                                           auth=self.keys, auth_dir=str(self.secrets))
        self.assertEqual((code, out), (0, "[]"))
        (seen,) = self.seen
        self.assertEqual(seen["argv"], ["/opt/modal/bin/modal", "app", "list"])
        self.assertEqual((seen["mode"], seen["uid"]), (0o600, os.getuid()))
        self.assertIn('token_id = "%s"' % TOKEN_ID, seen["text"])
        self.assertEqual(set(seen["env"]), {"PATH", "COLUMNS", "MODAL_CONFIG_PATH"})
        self.assert_no_secret(seen["argv"], seen["env"])
        self.assertEqual(self.files(), [])

    def test_removed_after_a_failure_a_timeout_and_an_error(self):
        code, _, _ = providers.run_modal(["app", "stop", "x"], env={}, runner=self.runner(3), auth=self.keys, auth_dir=str(self.secrets))
        self.assertEqual((code, self.files()), (3, []))
        with self.assertRaises(providers.ProviderError) as ctx:
            providers.run_modal(["deploy", "app.py"], env={}, runner=self.runner("timeout"), auth=self.keys, auth_dir=str(self.secrets))
        self.assertIn("timeout", str(ctx.exception))
        self.assertEqual(self.files(), [])
        with self.assertRaises(OSError):
            providers.run_modal(["app", "list"], env={}, runner=self.runner(OSError("exec")), auth=self.keys, auth_dir=str(self.secrets))
        self.assertEqual(self.files(), [])
        self.assertEqual([record["exists"] for record in self.seen], [True, True, True])
        self.assertEqual(len({record["path"] for record in self.seen}), 3)      # a file for each process

    def test_every_call_of_the_provider_passes_the_context(self):
        answers = iter(["[]", "[]", "", "", "[]"])
        files = []

        def run(argv, **kw):
            record = self.look(argv, kw["env"])
            files.append(record["path"])
            return subprocess.CompletedProcess(argv, 0, next(answers, "[]"), "")
        modal = providers.Modal("mdf-surya", runner=run, environ={"PATH": "/usr/bin"}, env_file=self.keys,
                                auth_dir=str(self.secrets))
        modal.app_containers()                                          # a list
        modal._json("app", "list")
        modal.system_logs(0)                                            # the logs of the system
        self.assertEqual([record["argv"][1:3] for record in self.seen], [["container", "list"], ["app", "list"], ["app", "logs"]])
        for record in self.seen:
            self.assertEqual(record.get("mode"), 0o600, record["argv"])
            self.assert_no_secret(record["argv"], record["env"])
        self.assertEqual(len(set(files)), len(files))
        self.assertEqual(self.files(), [])

    def test_with_the_runner_of_a_test_and_no_directory_no_file_is_written(self):
        """The existing tests of the provider give a keys file for the proxy token and a runner that starts
        nothing: without a real process there is nothing to protect, and they stay as they are."""
        modal = providers.Modal("mdf-surya", runner=self.runner(), environ={}, env_file=self.keys)
        modal._json("app", "list")
        self.assertEqual((self.seen[0]["path"], modal.auth, self.files()), (None, None, []))
        real = providers.Modal("mdf-surya", environ={}, env_file=self.keys)          # the real runner: always
        self.assertEqual((real.auth, real.auth_dir), (self.keys, "/run/mdf/secrets"))


class FakeProcess:
    def __init__(self, lines=()):
        self.stdout, self.returncode, self.pid = iter(lines), None, 4242

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15


class StreamTests(Base):
    def popen(self):
        def start(argv, **kw):
            self.look(argv, kw["env"])
            return FakeProcess()
        return start

    def source(self, **kw):
        return worker_logs.ModalLogs("mdf-surya", {"PATH": "/usr/bin"}, lambda: [], popen=self.popen(), **kw)

    def test_each_stream_has_its_own_file_until_its_process_ends(self):
        source = self.source(auth=self.keys, auth_dir=str(self.secrets))
        first, second = source.stream("app"), source.stream("system")
        self.assertIsInstance(first, worker_logs.AuthStream)
        self.assertEqual(len(self.files()), 2)
        providers.run_modal(["app", "list"], env={}, runner=self.runner(), auth=self.keys, auth_dir=str(self.secrets))
        paths = [record["path"] for record in self.seen]
        self.assertEqual(len(set(paths)), 3)                            # two streams and a synchronous call together
        for record in self.seen:
            self.assertEqual(record["mode"], 0o600)
            self.assert_no_secret(record["argv"], record["env"])
        self.assertEqual(len(self.files()), 2)                          # the call is over, the streams are not
        self.assertEqual(first.wait(), 0)
        self.assertEqual(self.files(), [os.path.basename(paths[1])])
        second.terminate()
        self.assertEqual(second.poll(), -15)                            # the follower stops it: seen by `poll`
        self.assertEqual(self.files(), [])
        self.assertEqual(first.pid, 4242)                               # everything else is the process's own

    def test_a_stream_that_cannot_start_leaves_no_file(self):
        def broken(argv, **kw):
            self.look(argv, kw["env"])
            raise OSError("exec")
        source = worker_logs.ModalLogs("mdf-surya", {}, lambda: [], popen=broken, auth=self.keys, auth_dir=str(self.secrets))
        with self.assertRaises(OSError):
            source.stream("app")
        self.assertEqual((self.seen[0]["exists"], self.files()), (True, []))

    def test_the_real_cli_does_not_follow_without_a_keys_file(self):
        source = worker_logs.ModalLogs("mdf-surya", {}, lambda: [])     # the real Popen, no `auth`
        with self.assertRaises(providers.ProviderError):
            source.stream("app")
        fake = self.source()                                            # a test without `auth`: as before
        self.assertIsInstance(fake.stream("app"), FakeProcess)
        self.assertIsNone(self.seen[0]["path"])

    def test_the_follower_of_a_session_passes_the_keys_file_of_the_worker(self):
        made = {}

        class Session:
            provider, resource, t_arm = "modal", "mdf-surya", 0.0

            class ep:
                key_env = ["MODAL_PROXY_TOKEN_ID"]

            class m:
                env_file = self.keys

        def logs(app, environ, containers, **kw):
            made["logs"] = kw
            return "source"

        def follower(source, path, t_arm):
            made["follower"] = (source, path)
            return "follower"
        with mock.patch.object(worker_logs, "ModalLogs", logs), mock.patch.object(worker_logs, "ModalFollower", follower), \
                mock.patch.object(providers, "Modal") as modal:
            self.assertEqual(lifecycle.real_follower(Session, self.root / "log"), "follower")
        self.assertEqual(made["logs"], {"auth": self.keys})
        self.assertEqual(modal.call_args.kwargs["env_file"], self.keys)
        # and the one of the provider itself (`log_follower`)
        mine = providers.Modal("mdf-surya", runner=self.runner(), environ={}, env_file=self.keys, auth_dir=str(self.secrets))
        with mock.patch.object(worker_logs, "ModalFollower", lambda source, path, t_arm, clock=None: source):
            source = mine.log_follower(self.root / "log", 0.0)
        self.assertEqual((source.auth, source.auth_dir), (self.keys, str(self.secrets)))

    def test_the_start_of_the_worker_sweeps_the_directory(self):
        import inspect
        self.assertIn("providers.modal_auth_sweep()", inspect.getsource(lifecycle.Manager.recover))


if __name__ == "__main__":
    unittest.main()
