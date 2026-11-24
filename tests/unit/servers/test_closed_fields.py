"""Closed fields in `session.py` and in the follower of the worker logs (MFL023 B1; SOL2-B2 of the plan r2, SOL3-N1
of the review of r3).

The text of the provider, carrying a synthetic key as it is and in every encoded form of `relay.credential_forms`, never
reaches the events, the report, the output of `main`, the ledger or the worker log saved by `Session._keep_log`: the
stderr of the CLI of Modal on the arm, the disarm, the zero and a failed `container list` of the follower, the stderr of
a log stream of the CLI, the body of a RunPod answer on PATCH and GET. The stderr of the CLI stays in the exception in
memory (the principle of `test_modal.py:203`); the real lines of the containers stay in the log. No network, no real
CLI of Modal: a fake runner and a fake `modal app logs -f`."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import providers as pv  # noqa: E402
import session  # noqa: E402
import worker_logs as wl  # noqa: E402

from markdown_for_llms.engine import envfile  # noqa: E402
from markdown_for_llms.engine.relay import credential_forms  # noqa: E402

KEY = "b1-synthetic-key/0123+456789=="
FORMS = sorted({f.decode() for f in credential_forms([KEY])})
LEAK = " ".join(FORMS)
TOKEN_ID = "wk-b1-token-id"
URL = "https://b1--mdf-surya-serve.modal.run"
CONTAINER = ("2026-11-24T10:00:01.000000+00:00 ta-01B1 INFO the stdout of the container",
             "2026-11-24T10:00:02.000000+00:00 ta-01B1 WARNING the stderr of the container")


def ok(out=""):
    return types.SimpleNamespace(returncode=0, stdout=out, stderr="")


def scratch(test):
    d = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, d, True)
    return d


def env_file(test):
    """The env file of the session: the synthetic key is a key of it, so the secret scan knows its plain form."""
    path = scratch(test) / ".env"
    path.write_text("MODAL_PROXY_TOKEN_ID=%s\nMODAL_PROXY_TOKEN_SECRET=%s\nRUNPOD_API_KEY=rk-b1-synthetic-0123\n"
                    % (TOKEN_ID, KEY))
    path.chmod(0o600)
    return path


class Clocks:
    def __init__(self):
        self.wall = 1_800_000_000.0

    def time(self):
        return self.wall

    def sleep(self, s):
        self.wall += s


class ModalCli:
    """The CLI of the SDK: every call answers, except the `nth` call of `fail` (`deploy`, `app stop`, `app list`,
    `container list`), whose stderr carries the key in all its forms."""

    def __init__(self, fail, nth=1):
        self.fail, self.nth, self.count = fail, nth, {}

    def __call__(self, argv, **kw):
        args = argv[len(pv.MODAL_SDK):]
        word = args[0] if args[0] == "deploy" else " ".join(args[:2])
        self.count[word] = self.count.get(word, 0) + 1
        if word == self.fail and (self.nth is None or self.count[word] == self.nth):
            return types.SimpleNamespace(returncode=1, stdout="", stderr="Error: request refused: token=" + LEAK + "\n")
        if word == "volume list":
            return ok(json.dumps([{"name": pv.VOLUME}]))
        if word == "workspace proxy-tokens":
            return ok(json.dumps([{"name": "p", "token_id": TOKEN_ID, "created_at": "x", "created_by": "y",
                                   "scoped": False}]))
        if word == "deploy":
            return ok("Created web function serve => %s\n" % URL)
        return ok("[]")


class Nowhere(unittest.TestCase):
    def assertNowhere(self, *texts):
        for text in texts:
            for form in FORMS:
                self.assertNotIn(form, text)


class TestEventsAndOutputs(Nowhere):
    """SOL2-B2: the arm, the disarm and the zero of Modal; the output of `main`; RunPod on PATCH and GET."""

    def modal_session(self, fail, nth=1):
        env = env_file(self)
        ledger = env.parent / "spend.jsonl"
        a = session.parser().parse_args(["run", "--provider", "modal", "--name", "mdf-surya", "--env-file", str(env),
                                         "--spend-ledger", str(ledger), "--spend-cap-usd", "3", "--cap-usd", "0.50",
                                         "--", "true"])
        a.command = a.command[1:] if a.command[:1] == ["--"] else a.command
        clocks, out = Clocks(), io.StringIO()
        m = pv.Modal("mdf-surya", runner=ModalCli(fail, nth), clock=clocks.time, sleep=clocks.sleep, environ={},
                     env_file=str(env))
        m.log_follower = None                      # the follower has its own test below
        ses = session.Session(a, m, clock=clocks.time, sleep=clocks.sleep, out=out)
        with mock.patch.object(session.subprocess, "Popen") as popen:
            popen.return_value.poll.return_value = 0
            popen.return_value.returncode = 0
            popen.return_value.pid = 4242
            code = ses.run()
        return code, ses, out.getvalue(), ledger.read_text()

    def test_the_arm_the_disarm_and_the_zero_of_modal(self):
        for fail, nth, event, op in (("deploy", 1, "provider-error", "modal.deploy"),
                                     ("app stop", 1, "disarm-error", "modal.app_stop"),
                                     ("app list", 2, "zero-error", "modal.app_list")):
            with self.subTest(fail=fail):
                code, ses, out, ledger = self.modal_session(fail, nth)
                errors = [e for e in ses.events if e["event"] == event]
                self.assertEqual(errors[0]["class"], "ProviderError")
                self.assertEqual(errors[0]["op"], op)
                self.assertIn('"kind": "disarm"', ledger)
                self.assertNowhere(json.dumps(ses.events), out, ledger)

    def test_the_text_of_the_cli_stays_in_the_exception(self):
        m = pv.Modal("mdf-surya", runner=ModalCli("app list"), environ={})
        with self.assertRaises(pv.ProviderError) as caught:
            m.state()
        exc = caught.exception
        self.assertIn(FORMS[-1], str(exc))                        # in memory only (test_modal.py:203)
        self.assertEqual((exc.local, exc.op), ("modal app list: exit 1", "modal.app_list"))
        self.assertEqual(pv.closed(exc), {"class": "ProviderError", "op": "modal.app_list", "http": None,
                                          "detail": "modal app list: exit 1"})

    def test_the_output_of_main(self):
        env = env_file(self)
        out = io.StringIO()

        def factory():
            return pv.Modal("mdf-surya", runner=ModalCli("app list", None), environ={}, env_file=str(env))
        code = session.main(["status", "--provider", "modal", "--name", "mdf-surya", "--env-file", str(env),
                             "--spend-ledger", str(env.parent / "spend.jsonl")], factory=factory, out=out)
        self.assertEqual(code, 4)
        line = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual((line["code"], line["class"], line["op"]), ("provider_error", "ProviderError", "modal.app_list"))
        self.assertNowhere(out.getvalue())

    def test_the_body_of_runpod_is_never_kept(self):
        def answer(request):
            return httpx.Response(500, text="refused " + LEAK)
        r = pv.RunPod(envfile.Secret("rk-b1-synthetic-0123"), "mdf-surya", transport=httpx.MockTransport(answer))
        for method in ("PATCH", "GET"):
            with self.subTest(method=method):
                with self.assertRaises(pv.ProviderError) as caught:
                    r.call(method, "/serverless/x")
                exc = caught.exception
                self.assertEqual((exc.op, exc.http, exc.remote), ("runpod." + method.lower(), 500, None))
                self.assertEqual(str(exc), "RunPod %s /serverless/x: HTTP 500" % method)
                self.assertNowhere(str(exc), json.dumps(pv.closed(exc)))


class FakeStream:
    """A `modal app logs -f`: the lines of the containers on its stdout; the text of the CLI on its stderr, which comes
    on the same pipe only when the caller merges the stderr into the stdout (`stderr=subprocess.STDOUT`)."""

    def __init__(self, lines, cli_stderr, merged):
        self.lines = list(lines) + (list(cli_stderr) if merged else [])
        self.done, self.returncode = threading.Event(), None

    @property
    def stdout(self):
        for x in self.lines:
            yield x + "\n"
        self.done.wait(10)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.done.wait(timeout if timeout is not None else 10)
        self.returncode = -15
        return self.returncode

    def terminate(self):
        self.done.set()

    def kill(self):
        self.done.set()


class TestTheFollowerOfModal(Nowhere):
    """SOL3-N1: a failed `container list` and the stderr of a log stream carry the key, as it is and encoded; neither
    reaches the worker log that `_keep_log` saves, and the real lines of the containers stay in it."""

    def keep(self, fail_list):
        env = env_file(self)
        logs_dir = scratch(self) / "worker-logs"
        a = session.parser().parse_args(["run", "--provider", "modal", "--name", "mdf-surya", "--env-file", str(env),
                                         "--spend-ledger", str(env.parent / "spend.jsonl"), "--spend-cap-usd", "3",
                                         "--cap-usd", "0.50", "--worker-log-dir", str(logs_dir), "--", "true"])
        m = pv.Modal("mdf-surya", runner=ModalCli("container list" if fail_list else None, None), environ={},
                     env_file=str(env))
        ses = session.Session(a, m, out=io.StringIO())
        cli = ["Warning: Received UNAUTHENTICATED token=" + LEAK + " for serve. Retrying..."]

        def popen(argv, **kw):
            source = argv[-1]
            lines = [CONTAINER[0]] if source == "stdout" else [CONTAINER[1]] if source == "stderr" else []
            return FakeStream(lines, cli, kw.get("stderr") is subprocess.STDOUT)
        tmp = Path(tempfile.mkdtemp(prefix="mdf-wlog-"))
        follower = wl.ModalFollower(wl.ModalLogs("mdf-surya", {}, m.app_containers, popen=popen),
                                    tmp / "worker-log.partial", time.time()).start()
        end = time.monotonic() + 10
        while not all(x.split(" ", 2)[2] in follower.path.read_text() for x in CONTAINER) or (
                fail_list and "follower_error" not in follower.path.read_text()):
            self.assertLess(time.monotonic(), end)
            time.sleep(0.05)
        info = ses.keep_log(follower, {"arm_id": "20261124T100000Z-b1b1b1b1"})
        return info, ses

    def test_a_failed_container_list_and_the_stderr_of_the_stream(self):
        for fail_list in (True, False):
            with self.subTest(fail_list=fail_list):
                info, ses = self.keep(fail_list)
                self.assertTrue(info.get("saved"), info)                # never withheld: the key never came in
                text = Path(info["saved"]).read_text()
                self.assertNowhere(text, json.dumps(ses.events))
                for line in CONTAINER:
                    self.assertIn(line.split(" ", 2)[2], text)          # the real logs of the container stay
                if fail_list:
                    error, = [wl.parse(x)["event"] for x in text.splitlines()
                              if wl.parse(x) and (wl.parse(x)["event"] or {}).get("event") == "follower_error"]
                    self.assertEqual((error["error"], error["op"], error["detail"]),
                                     ("ProviderError", "modal.container_list", "modal container list: exit 1"))

    def test_the_stderr_of_the_cli_goes_nowhere(self):
        seen = []

        def popen(argv, **kw):
            seen.append(kw)
            return FakeStream([], [], False)
        wl.ModalLogs("mdf-surya", {}, lambda: [], popen=popen).stream("system")
        self.assertIs(seen[0]["stderr"], subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
