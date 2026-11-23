"""Modal after the lessons of RunPod (MFL033 c7.8, prompt N4g §4.1): the state of the app with its
containers as workers, the Volume before every deploy, the CLI of `modal==1.6.3` (`--json` keys,
`stopping...`, `COLUMNS`), the follower of the logs and containers, and the drivers of
`modal_tasks.py` (readiness with the fast stop and the containers, `probe-negative`, `probe-valid`,
phase A). No network: a fake runner for the CLI, fake streams for `modal app logs`,
`httpx.MockTransport` for Modal's edge, a fake clock."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import drivers as dv  # noqa: E402
import modal_tasks as mt  # noqa: E402
import providers as pv  # noqa: E402
import worker_logs as wl  # noqa: E402

from markdown_for_llms.bench import serverspend, spend  # noqa: E402

URL = "https://ws--mdf-surya-serve.modal.run"
TOKEN_ID, TOKEN_SECRET = "wk-test-token-id", "ws-test-token-secret"


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def runner(answers, calls):
    """The CLI of the SDK: `answers` by the first two words after `modal` (a value, or a callable)."""
    def run(argv, **kw):
        calls.append((argv[len(pv.MODAL_SDK):], kw.get("env")))
        key = " ".join(argv[len(pv.MODAL_SDK):len(pv.MODAL_SDK) + 2])
        code, out = answers.get(key, (0, "[]"))
        return types.SimpleNamespace(returncode=code, stdout=out() if callable(out) else out, stderr="")
    return run


def app_row(state, app_id="ap-1", name="mdf-surya"):
    """A row of `modal app list --json` of modal 1.6.3 (`cli/app.py` list_, `_col_name_to_json_key`)."""
    return {"app_id": app_id, "description": name, "state": state, "tasks": "0", "created_at": "2026-11-19T13:40:00+01:00",
            "stopped_at": None}


def container_row(cid, start="Pending", app_id="ap-1", name="mdf-surya"):
    return {"container_id": cid, "app_id": app_id, "app_name": name, "start_time": start}


def token_row(token_id, name="mdf-surya-proxy", scoped=False):
    """A row of `modal workspace proxy-tokens list --json` of modal 1.6.3 (`cli/workspace.py`)."""
    return {"name": name, "token_id": token_id, "created_at": "2026-11-19T17:16:00+01:00", "created_by": "willermo",
            "scoped": scoped}


def env_file(test, text="MODAL_PROXY_TOKEN_ID=%s\nMODAL_PROXY_TOKEN_SECRET=%s\n" % (TOKEN_ID, TOKEN_SECRET)):
    tmp = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, tmp, True)
    path = tmp / ".env"
    path.write_text(text)
    path.chmod(0o600)
    return path


class TestModalProvider(unittest.TestCase):
    def test_state_of_the_sdk_with_the_containers_as_workers(self):
        calls = []
        apps = json.dumps([app_row("deployed"), app_row("deployed", "ap-2", "other")])
        conts = json.dumps([container_row("ta-1"), container_row("ta-2", "2026-11-19T13:41:07+01:00"),
                            container_row("ta-9", "2026-11-19T13:00:00+01:00", "ap-2", "other")])
        m = pv.Modal("mdf-surya", runner=runner({"app list": (0, apps), "container list": (0, conts)}, calls),
                     environ={"PATH": "/usr/bin"})
        s = m.state()
        self.assertEqual((s["armed"], s["busy"], s["containers"]), (True, True, ["ta-1", "ta-2"]))
        self.assertEqual(s["workers"], [{"id": "ta-1", "status": "PENDING", "start": None},
                                        {"id": "ta-2", "status": "RUNNING", "start": "2026-11-19T13:41:07+01:00"}])
        self.assertEqual([c["id"] for c in m.app_containers()], ["ta-1", "ta-2"])     # by `app_name`
        self.assertTrue(all(env["COLUMNS"] == pv.MODAL_COLUMNS for _, env in calls))

    def test_only_a_stopped_app_is_disarmed(self):
        for state, armed in (("deployed", True), ("stopping...", True), ("initializing...", True), ("stopped", False)):
            with self.subTest(state=state):
                m = pv.Modal("mdf-surya", runner=runner({"app list": (0, json.dumps([app_row(state)]))}, []), environ={})
                zero, s = m.zero()
                self.assertEqual((s["armed"], zero), (armed, not armed))

    def test_a_watch_sees_a_container_replaced(self):
        states = iter([[container_row("ta-1", "2026-11-19T13:41:07+01:00")], [container_row("ta-2")]])
        m = pv.Modal("mdf-surya", runner=runner({"app list": (0, json.dumps([app_row("deployed")])),
                                                 "container list": (0, lambda: json.dumps(next(states)))}, []),
                     environ={})
        w = dv.Watch(m, Clock(), Clock().sleep)
        w.read(no_restart=True)
        with self.assertRaisesRegex(dv.Violation, "a restart: worker ta-2"):
            w.read(no_restart=True)

    def test_no_deploy_without_the_volume(self):
        calls = []
        m = pv.Modal("mdf-surya", runner=runner({"volume list": (0, "[]")}, calls), environ={})
        with self.assertRaisesRegex(pv.ProviderError, "Volume mdf-surya-weights is missing"):
            m.arm()
        self.assertEqual([c[0][:2] for c in calls], [["volume", "list"]])

    def test_deploy_with_the_volume_the_fault_and_no_empty_value(self):
        calls = []
        out = "✓ Created web function serve.\n└── %s 🔑\n" % URL
        answers = {"volume list": (0, json.dumps([{"name": pv.VOLUME, "created_at": "x", "created_by": "y"}])),
                   "workspace proxy-tokens": (0, json.dumps([token_row(TOKEN_ID)])),
                   "deploy " + str(pv.HERE / "modal_app.py"): (0, out)}
        environ = {"PATH": "/usr/bin", "MDF_FAULT": "stale", "MODAL_PROXY_TOKEN_SECRET": "s", "RUNPOD_API_KEY": "k"}
        path = env_file(self)
        for fault in ("", "hang-after-first"):
            with self.subTest(fault=fault):
                del calls[:]
                m = pv.Modal("mdf-surya-fault", runner=runner(answers, calls), environ=environ, env_file=path)
                self.assertEqual(m.arm(fault), {"url": URL, "volume": pv.VOLUME, "proxy_token": {
                    "proxy_tokens": 1, "in_workspace": True, "scoped": False}})
                argv, env = calls[-1]
                self.assertEqual(argv[0], "deploy")
                self.assertEqual(env.get("MDF_FAULT"), fault or None)
                self.assertEqual((env["MDF_MODAL_APP"], env["COLUMNS"]), ("mdf-surya-fault", pv.MODAL_COLUMNS))
                self.assertFalse({"MODAL_PROXY_TOKEN_SECRET", "RUNPOD_API_KEY"} & set(env))

    def test_the_levers_of_the_deploy(self):
        """MFL033 N5b: MDF_LEVERS in the env of the deploy only with levers; a stale one never passes."""
        calls = []
        out = "✓ Created web function serve.\n└── %s 🔑\n" % URL
        answers = {"volume list": (0, json.dumps([{"name": pv.VOLUME, "created_at": "x", "created_by": "y"}])),
                   "workspace proxy-tokens": (0, json.dumps([token_row(TOKEN_ID)])),
                   "deploy " + str(pv.HERE / "modal_app.py"): (0, out)}
        environ = {"PATH": "/usr/bin", "MDF_LEVERS": "skip-mm-profiling"}
        m = pv.Modal("mdf-surya", runner=runner(answers, calls), environ=environ, env_file=env_file(self))
        m.arm()
        self.assertNotIn("MDF_LEVERS", calls[-1][1])
        m.arm("", False, "skip-mm-profiling")
        self.assertEqual(calls[-1][1]["MDF_LEVERS"], "skip-mm-profiling")
        del calls[:]
        with self.assertRaises(pv.ProviderError):
            m.arm("", False, "nope")
        self.assertEqual(calls, [])

    def test_the_system_lines_fetched_without_follow(self):
        """c7.14: one fetch of the system lines since the arm, never `-f` (the stream lost the
        scheduler's line on 20/11), whole seconds in UTC, the SDK's COLUMNS; an error raises."""
        calls = []
        fetched = "2026-11-20 09:09:07+01:00 fu-sF4Wg5rE4HlmgcKfKPvp5Q %s\n\n" % WAITING_TEXT
        m = pv.Modal("mdf-surya", runner=runner({"app logs": (0, fetched)}, calls),
                     environ={"PATH": "/usr/bin", "MODAL_PROXY_TOKEN_SECRET": "s"})
        self.assertEqual(m.system_logs(1795162037.3), [fetched.strip()])
        argv, env = calls[0]
        self.assertEqual(argv, ["app", "logs", "mdf-surya", "--since", "2026-11-20T08:07:17+00:00", "--tail", "1000",
                                "--timestamps", "--show-function-id", "--show-container-id", "-s", "system"])
        self.assertNotIn("-f", argv)
        self.assertEqual(env["COLUMNS"], pv.MODAL_COLUMNS)
        self.assertNotIn("MODAL_PROXY_TOKEN_SECRET", env)
        m = pv.Modal("mdf-surya", runner=runner({"app logs": (1, "")}, []), environ={})
        with self.assertRaisesRegex(pv.ProviderError, "^modal app logs: exit 1"):
            m.system_logs(1795162037)

    def test_the_inputs_of_the_web_function(self):
        """c7.16: the inputs of `serve` since a time, from `modal function calls APP/serve --since ISO
        --json` of 1.6.3 (the keys of `cli/function.py` calls), whole seconds in UTC; an error raises."""
        calls = []
        rows = [{"function_call_id": "fc-1", "service_method_name": None, "enqueued_at": "2026-11-20T09:28:52.8Z",
                 "started_at": "2026-11-20T09:28:52.9Z", "container_id": "ta-1", "startup_time_seconds": None,
                 "execution_time_seconds": 6.57, "status": "success"}, "not a row"]
        m = pv.Modal("mdf-surya", runner=runner({"function calls": (0, json.dumps(rows))}, calls),
                     environ={"PATH": "/usr/bin"})
        self.assertEqual(m.inputs(1795162037.9), rows[:1])
        argv, env = calls[0]
        self.assertEqual(argv, ["function", "calls", "mdf-surya/serve", "--since", "2026-11-20T08:07:17+00:00", "--json"])
        self.assertEqual(env["COLUMNS"], pv.MODAL_COLUMNS)
        m = pv.Modal("mdf-surya", runner=runner({"function calls": (1, "")}, []), environ={})
        with self.assertRaisesRegex(pv.ProviderError, "^modal function calls mdf-surya/serve"):
            m.inputs(1795162037)

    def test_the_rate_is_the_dearest_gpu_of_the_list(self):
        """c7.14 (prompt N4m §3.4): the Modal rate of the ledger and the drivers is the dearest GPU of
        the list plus 2 cores and 16 GiB, at the prices read on 20/11/2026."""
        from decimal import Decimal
        self.assertEqual(pv.MODAL_GPUS, ("L4", "A10G"))
        self.assertEqual(pv.RATES["modal"], Decimal("0.00036772"))              # A10G 0.000306 + 0.0000262 + 0.00003552
        self.assertEqual(pv.modal_rate(("L4",)), Decimal("0.00028372"))         # the rate of c7.13 and before
        self.assertEqual(pv.RATES["modal-cpu"], Decimal("0.00003508"))

    def test_an_error_of_the_cli_keeps_its_text(self):
        """c7.9: rich draws the error in a box as wide as COLUMNS (19/11, "Token missing"): the message
        of the error must keep the text, not the last 300 characters of the box."""
        width = int(pv.MODAL_COLUMNS) - 2
        panel = "╭─ Error %s╮\n│ Token missing. Could not authenticate client. %s│\n╰%s╯\n" % (
            "─" * (width - 8), " " * (width - 48), "─" * width)

        def run(argv, **kw):
            return types.SimpleNamespace(returncode=1, stdout="", stderr=panel)
        with self.assertRaisesRegex(pv.ProviderError, r"^modal app list: exit 1 Error Token missing\. Could not "
                                                      r"authenticate client\.$"):
            pv.Modal("mdf-surya", runner=run, environ={}).state()

    def test_nothing_to_stop_is_a_disarm(self):
        """c7.12: `modal app stop` exits 1 for an app never deployed or already stopped (modal 1.6.3
        `cli/app.py`): nothing is armed, so the disarm passes; any other error still raises."""
        def stop(code, err):
            calls = []

            def run(argv, **kw):
                calls.append(argv[len(pv.MODAL_SDK):])
                return types.SimpleNamespace(returncode=code, stdout="", stderr=err)
            m = pv.Modal("mdf-surya", runner=run, environ={})
            return m, calls

        for err in ("Error No App with name 'mdf-surya' found in the 'main' environment.\n",
                    "App is already stopped. (Stopped at 2026-11-19 15:43 by 'willermo').\n",
                    "╭─ Error ─╮\n│ No App with name 'mdf-surya' found in the 'main' environment. │\n╰─────╯\n"):
            with self.subTest(err=err[:30]):
                m, calls = stop(1, err)
                self.assertIsNone(m.disarm())
                self.assertEqual(calls, [["app", "stop", "mdf-surya", "--yes"]])
        for err in ("Error Token missing. Could not authenticate client.\n",
                    "Error No App with name 'mdf-surya-fault' found in the 'main' environment.\n", ""):
            with self.subTest(err=err[:30]):
                m, _ = stop(1, err)
                with self.assertRaisesRegex(pv.ProviderError, "^modal app stop: exit 1"):
                    m.disarm()

    def test_a_deploy_refused_closes_the_window_without_an_orphan(self):
        """c7.12, the case of 19/11: the SDK refuses the deploy before any app exists; the stop finds
        nothing, the state shows no app and no container, so the window closes (not clean) instead of
        staying orphan."""
        import session
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        ledger, env_file = tmp / "spend.jsonl", tmp / ".env"
        env_file.write_text("MODAL_PROXY_TOKEN_ID=wk-x-in-the-workspace\n")
        env_file.chmod(0o600)
        calls = []
        answers = {"volume list": (0, json.dumps([{"name": pv.VOLUME}]), ""),
                   "workspace proxy-tokens": (0, json.dumps([token_row("wk-x-in-the-workspace")]), ""),
                   "deploy": (1, "", "InvalidError: Web Functions do not support retries.\n"),
                   "app stop": (1, "", "Error No App with name 'mdf-surya' found in the 'main' environment.\n"),
                   "app list": (0, "[]", ""), "container list": (0, "[]", "")}

        def run(argv, **kw):
            args = argv[len(pv.MODAL_SDK):]
            calls.append(args[:2])
            code, out, err = answers["deploy" if args[0] == "deploy" else " ".join(args[:2])]
            return types.SimpleNamespace(returncode=code, stdout=out, stderr=err)

        a = session.parser().parse_args(["run", "--provider", "modal", "--name", "mdf-surya", "--env-file", str(env_file),
                                         "--spend-ledger", str(ledger), "--spend-cap-usd", "15", "--cap-usd", "0.30",
                                         "--", "true"])
        if a.command and a.command[0] == "--":
            a.command = a.command[1:]
        clock, out = Clock(), io.StringIO()
        code = session.Session(a, pv.Modal("mdf-surya", runner=run, environ={}, env_file=str(env_file)), clock=clock,
                               sleep=clock.sleep, out=out).run()
        self.assertEqual(code, 4)                                              # the arm failed
        self.assertIn(["deploy", str(pv.HERE / "modal_app.py")], calls)       # the token passed: the SDK refused
        self.assertEqual(calls.count(["app", "stop"]), 1)
        rows = spend.read(ledger)
        disarm = [r for r in rows if r.get("kind") == "disarm"]
        self.assertEqual(len(disarm), 1)
        self.assertFalse(disarm[0]["clean"])
        self.assertFalse(disarm[0].get("orphan"))
        self.assertEqual(serverspend.summary(rows, "modal", ledger)["orphans"], [])

    def test_a_deploy_past_the_term_stops_the_app_without_the_command(self):
        """S-B4 of the final review of MFL033: the deploy gets at most the time left to T; a deploy that
        ends after T (it returned, or the CLI was stopped at its timeout) is an uncertain arm: no command,
        the app stopped, the window closed (not clean), exit 5."""
        import session
        import subprocess
        for how in ("returned", "timeout"):
            with self.subTest(how=how):
                tmp = Path(tempfile.mkdtemp())
                self.addCleanup(shutil.rmtree, tmp, True)
                ledger, env_file = tmp / "spend.jsonl", tmp / ".env"
                env_file.write_text("MODAL_PROXY_TOKEN_ID=wk-x-in-the-workspace\n")
                env_file.chmod(0o600)
                clock, out, calls, timeouts = Clock(), io.StringIO(), [], {}
                answers = {"volume list": (0, json.dumps([{"name": pv.VOLUME}]), ""),
                           "workspace proxy-tokens": (0, json.dumps([token_row("wk-x-in-the-workspace")]), ""),
                           "app stop": (0, "", ""), "app list": (0, "[]", ""), "container list": (0, "[]", "")}

                def run(argv, **kw):
                    args = argv[len(pv.MODAL_SDK):]
                    calls.append(args[:2])
                    timeouts.setdefault(args[0] if args[0] == "deploy" else " ".join(args[:2]), []).append(kw["timeout"])
                    if args[0] == "deploy":
                        clock.t += 2000                    # a slow deploy, past T
                        if how == "timeout":
                            raise subprocess.TimeoutExpired(argv, kw["timeout"])
                        return types.SimpleNamespace(returncode=0, stdout="Created web function serve => %s\n" % URL,
                                                     stderr="")
                    code, o, err = answers[" ".join(args[:2])]
                    return types.SimpleNamespace(returncode=code, stdout=o, stderr=err)

                a = session.parser().parse_args(["run", "--provider", "modal", "--name", "mdf-surya", "--env-file",
                                                 str(env_file), "--spend-ledger", str(ledger), "--spend-cap-usd", "15",
                                                 "--cap-usd", "0.30", "--", "true"])
                a.command = a.command[1:] if a.command[:1] == ["--"] else a.command
                ses = session.Session(a, pv.Modal("mdf-surya", runner=run, clock=clock, environ={},
                                                  env_file=str(env_file)), clock=clock, sleep=clock.sleep, out=out)
                with mock.patch.object(session.subprocess, "Popen") as popen:
                    code = ses.run()
                self.assertEqual(code, 5)
                popen.assert_not_called()
                arm, = [r for r in spend.read(ledger) if r.get("kind") == "arm"]
                window = float(arm["until"]) - float(arm["t_arm"])
                self.assertLess(window, 1200)
                self.assertEqual(timeouts["deploy"], [window])            # the time left, not the 1200 s
                self.assertTrue(all(t <= 120 for t in timeouts["volume list"] + timeouts["workspace proxy-tokens"]))
                self.assertEqual(calls.count(["app", "stop"]), 1)        # the verification and the disarm
                disarm, = [r for r in spend.read(ledger) if r.get("kind") == "disarm"]
                self.assertFalse(disarm["clean"])
                detail, = [e["detail"] for e in ses.events if e["event"] == "provider-error"]
                self.assertIn("term of the session", detail)

    def test_the_follower_of_a_modal_app(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        f = pv.Modal("mdf-surya", runner=runner({}, []), environ={"PATH": "/usr/bin", "RUNPOD_API_KEY": "k"}) \
            .log_follower(tmp / "log", 1.0)
        self.assertIsInstance(f, wl.ModalFollower)
        self.assertEqual(f.source.argv[len(pv.MODAL_SDK):],
                         ["app", "logs", "mdf-surya", "-f", "--timestamps", "--show-container-id", "-s"])
        self.assertNotIn("RUNPOD_API_KEY", f.source.env)
        f.fh.close()


class TestProxyToken(unittest.TestCase):
    """c7.13: before the deploy, the proxy token of the env file must be among the proxy tokens of the
    workspace (19/11: `modal workspace proxy-tokens list` gave `[]` in `willermo`, and the edge refused
    every request of `probe-valid`). The id is compared in memory and never printed."""

    VOLUME_OK = (0, json.dumps([{"name": pv.VOLUME}]))
    OUT = "✓ Created web function serve.\n└── %s 🔑\n" % URL

    def arm(self, tokens, path=None, environ=None):
        calls = []
        answers = {"volume list": self.VOLUME_OK, "workspace proxy-tokens": (0, json.dumps(tokens)),
                   "deploy " + str(pv.HERE / "modal_app.py"): (0, self.OUT)}
        m = pv.Modal("mdf-surya", runner=runner(answers, calls), environ=environ or {"PATH": "/usr/bin"},
                     env_file=env_file(self) if path is None else path)
        return m, calls

    def test_no_deploy_when_the_token_is_not_in_the_workspace(self):
        for name, tokens in (("no token", []), ("other tokens", [token_row("wk-other-token-one"),
                                                                 token_row("wk-other-token-two", scoped=True)])):
            with self.subTest(name):
                m, calls = self.arm(tokens)
                with self.assertRaisesRegex(pv.ProviderError, "proxy_token_not_in_workspace") as cm:
                    m.arm()
                self.assertIn("among the %d proxy tokens of the workspace: no deploy" % len(tokens), str(cm.exception))
                self.assertNotIn(TOKEN_ID, str(cm.exception))
                self.assertNotIn("wk-other", str(cm.exception))
                self.assertEqual([c[0] for c in calls], [["volume", "list", "--json"],
                                                         ["workspace", "proxy-tokens", "list", "--json"]])

    def test_the_token_in_the_workspace_is_never_printed(self):
        m, calls = self.arm([token_row("wk-other-token-one"), token_row(TOKEN_ID, scoped=True)])
        info = m.arm()
        self.assertEqual(info["proxy_token"], {"proxy_tokens": 2, "in_workspace": True, "scoped": True})
        self.assertEqual(calls[-1][0][0], "deploy")
        self.assertNotIn(TOKEN_ID, json.dumps(info))
        for argv, env in calls:                                   # neither in an argv nor in an env of the CLI
            self.assertNotIn(TOKEN_ID, " ".join(argv))
            self.assertNotIn(TOKEN_ID, json.dumps(env))
        self.assertEqual(m.proxy_token(), info["proxy_token"])

    def test_no_deploy_without_the_env_file_or_the_key(self):
        m, calls = self.arm([token_row(TOKEN_ID)])
        m.env_file = None
        with self.assertRaisesRegex(pv.ProviderError, "no env file, the proxy token cannot be checked"):
            m.arm()
        m, calls = self.arm([token_row(TOKEN_ID)], path=env_file(self, "MODAL_PROXY_TOKEN_SECRET=%s\n" % TOKEN_SECRET))
        with self.assertRaisesRegex(pv.ProviderError, "MODAL_PROXY_TOKEN_ID missing in the env file"):
            m.arm()
        self.assertNotIn("deploy", [c[0][0] for c in calls])

    def test_an_error_of_the_cli_is_no_deploy(self):
        calls = []

        def run(argv, **kw):
            args = argv[len(pv.MODAL_SDK):]
            calls.append(args)
            if args[:2] == ["workspace", "proxy-tokens"]:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="Error Token missing.\n")
            return types.SimpleNamespace(returncode=0, stdout=self.VOLUME_OK[1], stderr="")
        m = pv.Modal("mdf-surya", runner=run, environ={}, env_file=env_file(self))
        with self.assertRaisesRegex(pv.ProviderError, "^modal workspace proxy-tokens list: exit 1 Error Token missing"):
            m.arm()
        self.assertNotIn("deploy", [c[0] for c in calls])

    def test_make_gives_the_env_file(self):
        path = env_file(self)
        self.assertEqual(pv.make("modal", "mdf-surya", str(path)).env_file, str(path))

    def test_the_session_refuses_the_arm_without_a_deploy(self):
        """The case of 19/11 before any request: the window closes at once, not orphan, with the
        refusal in the events of the session; the id is nowhere in its output or in the ledger."""
        import session
        path = env_file(self)
        ledger = path.parent / "spend.jsonl"
        calls = []
        answers = {"volume list": self.VOLUME_OK, "workspace proxy-tokens": (0, json.dumps([token_row("wk-other-token")])),
                   "app stop": (1, ""), "app list": (0, "[]"), "container list": (0, "[]")}

        def run(argv, **kw):
            args = argv[len(pv.MODAL_SDK):]
            calls.append(args[:2])
            code, out = answers.get(" ".join(args[:2]), (0, self.OUT))
            err = "Error No App with name 'mdf-surya' found in the 'main' environment.\n" if code else ""
            return types.SimpleNamespace(returncode=code, stdout=out, stderr=err)

        a = session.parser().parse_args(["run", "--provider", "modal", "--name", "mdf-surya", "--env-file", str(path),
                                         "--spend-ledger", str(ledger), "--spend-cap-usd", "15", "--cap-usd", "0.30",
                                         "--", "true"])
        a.command = a.command[1:] if a.command[:1] == ["--"] else a.command
        clock, out = Clock(), io.StringIO()
        code = session.Session(a, pv.Modal("mdf-surya", runner=run, environ={}, env_file=str(path)), clock=clock,
                               sleep=clock.sleep, out=out).run()
        self.assertEqual(code, 4)
        self.assertNotIn("deploy", [c[0] for c in calls])
        report = json.loads(out.getvalue().splitlines()[-1])
        errors = [e["detail"] for e in report["events"] if e["event"] == "provider-error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("proxy_token_not_in_workspace", errors[0])
        disarm = [r for r in spend.read(ledger) if r.get("kind") == "disarm"]
        self.assertEqual((len(disarm), disarm[0]["clean"]), (1, False))
        self.assertFalse(disarm[0].get("orphan"))
        self.assertNotIn(TOKEN_ID, out.getvalue() + ledger.read_text())


class TestModalApp(unittest.TestCase):
    """modal_app.py never creates the Volume and passes MDF_FAULT only with a fault; both apps build
    the vLLM image with `python` on PATH, the same way (c7.10)."""

    def load(self, environ, script="modal_app.py"):
        seen = {"registry": []}

        class App:
            def __init__(self, name):
                seen["app"] = name

            def function(self, **kw):
                seen["function"] = kw
                return lambda f: f

        image = mock.Mock()
        image.entrypoint.return_value = image
        image.add_local_file.return_value = image
        fake = types.ModuleType("modal")
        fake.Image = types.SimpleNamespace(
            from_registry=lambda ref, **kw: seen["registry"].append((ref, kw)) or image)
        fake.Volume = types.SimpleNamespace(from_name=lambda name, **kw: seen.setdefault("volume", (name, kw)))
        fake.App = App
        fake.concurrent = lambda **kw: (lambda f: f)
        fake.web_server = lambda **kw: (lambda f: f)
        with mock.patch.dict(sys.modules, {"modal": fake}), mock.patch.dict("os.environ", environ, clear=False):
            for k in ("MDF_FAULT", "MDF_MODAL_APP", "MDF_LEVERS"):
                if k not in environ:
                    __import__("os").environ.pop(k, None)
            spec = importlib.util.spec_from_file_location("mdf_modal_app_t", ROOT / "scripts" / "servers" / script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        seen["chain"] = [c for c in image.mock_calls if c[0] in ("entrypoint", "add_local_file")]
        return mod, seen

    def test_python_on_path_in_both_images(self):
        """Modal refuses a registry image without `python` on PATH (19/11): the vLLM image has only
        `python3`, so both apps add the link to its Python 3.12 and build the same image."""
        _, gpu = self.load({}, "modal_app.py")
        _, cpu = self.load({}, "modal_populate.py")
        self.assertEqual(len(gpu["registry"]), 1)
        ref, kw = gpu["registry"][0]
        self.assertEqual(ref, pv.IMAGE)
        self.assertEqual(kw, {"setup_dockerfile_commands": ["RUN ln -s /usr/bin/python3.12 /usr/local/bin/python"]})
        self.assertNotIn("add_python", kw)                  # the image's own Python, with vLLM and huggingface_hub
        self.assertEqual(cpu["registry"], gpu["registry"])
        self.assertEqual(cpu["chain"], gpu["chain"])

    def test_volume_and_fault(self):
        mod, seen = self.load({"MDF_MODAL_APP": "mdf-surya-fault"})
        self.assertEqual(seen["volume"], ("mdf-surya-weights", {"create_if_missing": False}))
        self.assertNotIn("MDF_FAULT", mod.ENV)
        self.assertEqual((seen["app"], seen["function"]["max_containers"]), ("mdf-surya-fault", 1))
        mod, _ = self.load({"MDF_FAULT": "crash-after-first"})
        self.assertEqual(mod.ENV["MDF_FAULT"], "crash-after-first")
        self.assertNotIn("MDF_LEVERS", mod.ENV)
        mod, _ = self.load({"MDF_LEVERS": "skip-mm-profiling"})
        self.assertEqual(mod.ENV["MDF_LEVERS"], "skip-mm-profiling")

    def test_the_volume_committed_once_the_cache_is_written(self):
        """MFL033 c8.4: the populate of the cache of torch.compile commits the Volume after the watchdog
        wrote the archive (the marker), with a line `mdf_compile_cache` phase commit; nothing without it."""
        import contextlib
        import io as _io
        import tempfile as _tempfile
        mod, _ = self.load({})
        mod.weights = mock.Mock()
        marker = Path(_tempfile.mkdtemp()) / "written"
        self.addCleanup(__import__("shutil").rmtree, marker.parent, True)
        slept = []
        err = _io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertFalse(mod.commit_when_written(str(marker), limit_s=0.0, sleep=slept.append))
        mod.weights.commit.assert_not_called()
        marker.write_text("/weights/mdf-compile-cache/torch-compile-x.tar")
        with contextlib.redirect_stderr(err):
            self.assertTrue(mod.commit_when_written(str(marker), limit_s=5, sleep=slept.append))
        mod.weights.commit.assert_called_once_with()
        line = json.loads(err.getvalue().splitlines()[-1])
        self.assertEqual((line["event"], line["phase"], line["ok"]), ("mdf_compile_cache", "commit", True))

    def test_an_l4_else_an_a10g(self):
        """c7.14: `gpu` is the list L4, A10G in this order (modal 1.6.3 ranks a `list`, refuses a
        tuple), the same as `providers.MODAL_GPUS` that sets the rate; cores and memory unchanged."""
        for app in ("mdf-surya", "mdf-surya-fault"):
            with self.subTest(app=app):
                mod, seen = self.load({"MDF_MODAL_APP": app})
                fn = seen["function"]
                self.assertIs(type(fn["gpu"]), list)
                self.assertEqual(fn["gpu"], ["L4", "A10G"])
                self.assertEqual(fn["gpu"], list(pv.MODAL_GPUS))
                self.assertEqual((fn["cpu"], fn["memory"]), ((2.0, 2.0), (16384, 16384)))

    def test_no_retry_policy_on_the_web_function(self):
        """c7.12: modal 1.6.3 refuses a retry policy on a web function, `retries=0` included (19/11, the
        first deploy); without the argument it has none (`_parse_retries(None)`)."""
        _, seen = self.load({})
        self.assertNotIn("retries", seen["function"])
        self.assertEqual((seen["function"]["min_containers"], seen["function"]["max_containers"]), (0, 1))


class FakeProc:
    """A `modal app logs -f`: its lines, then the end (`end`) or a wait until terminated."""

    def __init__(self, lines, end=False):
        self.lines, self.end, self.done = list(lines), end, threading.Event()
        self.terminated = False
        self.returncode = None

    @property
    def stdout(self):
        for x in self.lines:
            yield x + "\n"
        if not self.end:
            self.done.wait(5)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self.end:
            self.done.wait(timeout if timeout is not None else 5)
        self.returncode = 0 if self.end else -15
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.done.set()

    def kill(self):
        self.terminate()


class FakeLogs:
    def __init__(self, streams, containers):
        self.streams, self.containers_seq, self.started = streams, list(containers), []

    def stream(self, source):
        self.started.append(source)
        procs = self.streams.get(source) or [FakeProc([])]
        return procs.pop(0) if len(procs) > 1 else procs[0]

    def containers(self):
        return self.containers_seq.pop(0) if len(self.containers_seq) > 1 else self.containers_seq[0]


START = '{"event": "mdf_start", "t": 1.0, "mono": 1.0, "provider": "modal"}'
VLLM_ERROR = "(APIServer pid=12) ERROR 11-19 13:45:01 [core.py:123] EngineCore failed to start."


class TestModalFollower(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = self.dir / "worker-log.partial"

    def follow(self, source):
        f = wl.ModalFollower(source, self.path, time.time() - 10, list_every=0.02)
        f.retry_s = 0.02
        return f.start()

    def rows(self):
        return [wl.parse(x) for x in self.path.read_text().splitlines()]

    def test_lines_sources_containers_and_a_stream_started_again(self):
        ts = "2026-11-19 13:45:00+01:00"
        stdout = [FakeProc(["%s ta-01ABC %s" % (ts, VLLM_ERROR)], end=True),
                  FakeProc(["%s ta-01ABC %s" % (ts, VLLM_ERROR), "%s ta-01ABC second" % ts])]
        logs = FakeLogs({"stdout": stdout, "stderr": [FakeProc(["%s ta-01ABC %s" % (ts, START)])],
                         "system": [FakeProc(["%s ta-01ABC Container started" % ts, "not a log line"])]},
                        [[], [{"id": "ta-01ABC", "status": "PENDING", "start": None}],
                         [{"id": "ta-01ABC", "status": "RUNNING", "start": "13:45:00"}], []])
        f = self.follow(logs)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "second" not in self.path.read_text():
            time.sleep(0.02)
        time.sleep(0.2)
        info = f.stop()
        rows = self.rows()
        lines = [(r["worker"], r["source"], r["ts"], r["line"]) for r in rows if r["source"] != "session"]
        self.assertEqual(lines.count(("ta-01ABC", "container", ts, VLLM_ERROR)), 1)      # not repeated
        self.assertIn(("ta-01ABC", "container", ts, "second"), lines)
        self.assertIn(("ta-01ABC", "container", ts, START), lines)
        self.assertIn(("ta-01ABC", "system", ts, "Container started"), lines)
        self.assertIn(("-", "system", "", "not a log line"), lines)
        events = [json.loads(r["line"]) for r in rows if r["source"] == "session"]
        self.assertEqual([(e.get("status"), e.get("start")) for e in events if e.get("event") == "worker"],
                         [("PENDING", None), ("RUNNING", "13:45:00"), ("GONE", None)])
        self.assertTrue(any(e.get("event") == "logs_stream_end" and e.get("source") == "stdout" for e in events))
        self.assertEqual(info["workers"], ["ta-01ABC"])
        self.assertTrue(all(p.terminated for p in stdout[1:] + logs.streams["stderr"] + logs.streams["system"]))
        self.assertEqual(oct(self.path.stat().st_mode & 0o777), "0o600")
        rules = wl.StartRules()                       # the fast stop reads the Modal lines like RunPod's
        self.assertIn("an error of vLLM", rules.feed(self.path) or "")

    def test_the_cli_streams_are_run_with_the_sdk(self):
        seen = []

        def popen(argv, **kw):
            seen.append((argv, kw))
            return FakeProc([], end=True)
        logs = wl.ModalLogs("mdf-surya", {"PATH": "/usr/bin"}, lambda: [], popen=popen)
        logs.stream("stderr")
        argv, kw = seen[0]
        self.assertEqual(argv, pv.MODAL_SDK + ["app", "logs", "mdf-surya", "-f", "--timestamps", "--show-container-id",
                                               "-s", "stderr"])
        self.assertEqual((kw["env"]["COLUMNS"], kw["stderr"]), (pv.MODAL_COLUMNS, __import__("subprocess").STDOUT))


class StateOnly:
    """The readings of `providers.Modal`: `state()` from `states`, `system_logs(since)` from `system`
    (a callable of `since`, or none: no system line; c7.14)."""

    def __init__(self, states, system=None, inputs=None):
        self.states = list(states)
        self.system, self.fetches = system, []
        self.input_rows, self.input_since = inputs, []

    def state(self):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def system_logs(self, since):
        self.fetches.append(since)
        return self.system(since) if self.system else []

    def inputs(self, since):
        """`modal function calls` (c7.16): the rows given, or the exception given."""
        self.input_since.append(since)
        if isinstance(self.input_rows, Exception):
            raise self.input_rows
        return list(self.input_rows or [])


def mstate(*containers, armed=True):
    """A reading of `providers.Modal.state()`: (id, status) pairs as containers and workers."""
    return {"armed": armed, "busy": bool(containers), "apps": [], "containers": [c for c, _ in containers],
            "workers": [{"id": c, "status": s, "start": None} for c, s in containers]}


class Tasks(unittest.TestCase):
    """The drivers of `modal_tasks.py` as the session runs them: the arm row, MDF_SESSION_ARM, the log."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.env = self.dir / ".env"
        self.env.write_text("MODAL_PROXY_TOKEN_ID=%s\nMODAL_PROXY_TOKEN_SECRET=%s\n" % (TOKEN_ID, TOKEN_SECRET))
        self.env.chmod(0o600)
        self.servers = self.dir / "servers.json"
        self.servers.write_text(json.dumps({"default": "modal-l4", "servers": {"modal-l4": {
            "kind": "modal", "url": URL, "rate_usd_per_s": "0.00028372", "gpu": "L4"}}}))
        self.log = self.dir / "worker-log.partial"
        self.log.write_text("")
        self.ledger = self.dir / "spend.jsonl"
        self.clock = Clock()
        t_arm = self.clock.t - 20
        with spend.Ledger.file(self.ledger).locked() as locked:
            self.arm_id = serverspend.arm(locked, self.ledger, "modal", "mdf-surya", URL, "60", t_arm + 1935, "1500",
                                          pv.RATES["modal"], t_arm)["arm_id"]
        self.calls = []

    def edge(self, routes):
        """Modal's edge: 401 without the proxy token, else the answers of the proxy by (method, path)."""
        def handle(request):
            self.calls.append((request.method, request.url.path))
            if (request.headers.get("modal-key"), request.headers.get("modal-secret")) != (TOKEN_ID, TOKEN_SECRET):
                return httpx.Response(401, text="modal-http: invalid credentials for proxy authorization")
            answer = routes.get((request.method, request.url.path), (404, {}, None))
            answer = answer() if callable(answer) else answer
            if isinstance(answer, Exception):
                raise answer
            status, body, proxy = answer
            headers = {"x-mdf-proxy": proxy} if proxy else {}
            if isinstance(body, str):
                return httpx.Response(status, text=body, headers=headers)
            return httpx.Response(status, json=body, headers=headers)
        return httpx.MockTransport(handle)

    def tasks(self, argv, states, routes, log=True, system=None, inputs=None):
        a = mt.parser().parse_args(["--env-file", str(self.env), "--server", "modal-l4", "--servers", str(self.servers)]
                                   + argv + ["--app", "mdf-surya", "--spend-ledger", str(self.ledger)])
        environ = {serverspend.SESSION_ENV: self.arm_id}
        if log:
            environ[wl.LOG_ENV] = str(self.log)
        self.modal = StateOnly(states, system, inputs)
        return mt.Tasks(a, modal=self.modal, transport=self.edge(routes), clock=self.clock,
                        sleep=self.clock.sleep, environ=environ)

    def run_quiet(self, fn):
        out = io.StringIO()
        with redirect_stdout(out):
            code = fn()
        return code, [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]

    def write_log(self, *lines):
        with open(self.log, "a") as fh:
            for line in lines:
                fh.write("2026-11-19T12:45:00.000Z\t+30.0\tta-1\tcontainer\t2026-11-19 13:45:00+01:00\t%s\n" % line)

    def models(self, held):
        calls = []

        def answer():
            calls.append(1)
            if len(calls) <= held:
                self.clock.t += 30.0
                return httpx.ReadTimeout("held by Modal")
            return (200, {"data": [{"id": dv.MODEL}]}, "forwarded")
        return answer

    def valid_routes(self, held=3, version="0.25.1"):
        return {("GET", "/v1/models"): self.models(held), ("GET", "/version"): (200, {"version": version}, "forwarded"),
                ("GET", "/metrics"): (200, {}, "forwarded"), ("GET", "/health"): (200, {}, "health"),
                ("GET", "/ping"): (200, {}, "health"),
                ("POST", "/v1/chat/completions"): (200, {"choices": []}, "forwarded")}


class TestProbeNegative(Tasks):
    def test_passes_with_401_of_the_edge_and_no_container(self):
        code, lines = self.run_quiet(self.tasks(["probe-negative"], [mstate()], {}).probe_negative)
        res = lines[-1]
        self.assertEqual(code, 0, res)
        self.assertEqual(res["probe_negative"], {"no_key": {"status": 401, "proxy": None, "error": None},
                                                 "wrong_key": {"status": 401, "proxy": None, "error": None}})
        self.assertEqual((res["violation"], res["containers"], res["proxy_requests"]), (None, [], []))
        self.assertGreaterEqual(len(res["readings"]), 13)                  # 120 s, a reading every 10 s

    def test_fails(self):
        cases = (("a container", [mstate(), mstate(("ta-1", "PENDING"))], None, True, "a worker or container"),
                 ("no log", [mstate()], None, False, "no log of the session"),
                 ("a request in the log", [mstate()], '{"event": "mdf_request", "kind": "error", "status": 502}', True,
                  "a request answered by the proxy"))
        for name, states, line, log, expected in cases:
            with self.subTest(name):
                self.log.write_text("")
                if line:
                    self.write_log(line)
                code, lines = self.run_quiet(self.tasks(["probe-negative"], states, {}, log=log).probe_negative)
                self.assertEqual(code, 6)
                self.assertIn(expected, lines[-1]["violation"])

    def test_an_answer_of_the_proxy_is_not_the_edge(self):
        t = self.tasks(["probe-negative"], [mstate()], {})

        def proxy(request):
            return httpx.Response(401, json={}, headers={"x-mdf-proxy": "error"})
        t.transport = httpx.MockTransport(proxy)
        code, lines = self.run_quiet(t.probe_negative)
        self.assertEqual(code, 6)
        self.assertIn("not a 401 of the edge", lines[-1]["violation"])
        self.assertEqual(lines[-1]["readings"], [])                      # stopped before any reading


class TestProbeValid(Tasks):
    def test_passes_with_the_times_from_the_arm(self):
        states = [mstate(), mstate(("ta-1", "PENDING")), mstate(("ta-1", "RUNNING"))]
        code, lines = self.run_quiet(self.tasks(["probe-valid"], states, self.valid_routes(held=3)).probe_valid)
        res = lines[-1]
        self.assertEqual(code, 0, res)
        ready = res["probe_valid"]["ready"]
        self.assertEqual((ready["ready"], ready["first_send_since_arm_s"], ready["seconds"]), (True, 20.0, 105.0))
        self.assertEqual(ready["since_arm_s"], 125.0)
        self.assertEqual((res["probe_valid"]["ping"]["proxy"], res["probe_valid"]["chat"]["proxy"]), ("health", "forwarded"))
        self.assertEqual((res["containers"], res["max_total"]), (["ta-1"], 1))
        self.assertGreaterEqual(res["readings"], ready["readings"] + 3)    # 30 s after the chat

    def test_the_fast_stop_before_readiness(self):
        cases = (("an error of vLLM", [mstate(("ta-1", "RUNNING"))], VLLM_ERROR, "an error of vLLM"),
                 ("the exit of vLLM", [mstate(("ta-1", "RUNNING"))], '{"event": "mdf_vllm_exit", "code": 1}',
                  "mdf_vllm_exit"),
                 ("two containers", [mstate(), mstate(("ta-1", "RUNNING"), ("ta-2", "PENDING"))], None, "total 2 > 1"),
                 ("a container replaced", [mstate(("ta-1", "RUNNING")), mstate(), mstate(("ta-2", "PENDING"))], None,
                  "a restart"))
        for name, states, line, expected in cases:
            with self.subTest(name):
                self.log.write_text("")
                if line:
                    self.write_log(line)
                self.calls = []
                code, lines = self.run_quiet(self.tasks(["probe-valid"], states,
                                                        self.valid_routes(held=10 ** 6)).probe_valid)
                self.assertEqual(code, 6)
                self.assertIn(expected, lines[-1]["probe_valid"]["ready"]["stopped"])
                self.assertNotIn(("POST", "/v1/chat/completions"), self.calls)

    def test_a_wrong_version_or_a_second_container_after_the_chat(self):
        code, lines = self.run_quiet(self.tasks(["probe-valid"], [mstate(("ta-1", "RUNNING"))],
                                                self.valid_routes(held=0, version="0.11.0")).probe_valid)
        self.assertEqual((code, lines[-1]["pass"]), (6, False))
        states = [mstate(("ta-1", "RUNNING")), mstate(("ta-1", "RUNNING"), ("ta-2", "PENDING"))]
        code, lines = self.run_quiet(self.tasks(["probe-valid"], states, self.valid_routes(held=0)).probe_valid)
        self.assertEqual(code, 6)
        self.assertIn("total 2 > 1", lines[-1]["violation"])


class TestPhaseA(Tasks):
    def phase_a(self, routes, states):
        with mock.patch.object(dv.Inference, "abandon", lambda self, after_s: None):     # never the network
            return self.run_quiet(self.tasks(["fault-test", "--phase", "A"], states, routes).fault_test)

    def test_the_first_request_must_succeed(self):
        routes = {**self.valid_routes(held=0), ("POST", "/v1/chat/completions"): (500, {}, "forwarded")}
        code, lines = self.phase_a(routes, [mstate(("ta-1", "RUNNING"))])
        self.assertEqual((code, lines[-1]["violation"]), (6, "the first request failed"))

    def test_a_held_answer_that_is_not_json_is_a_violation(self):
        chats = iter([(200, {"choices": []}, "forwarded")])         # then a JSON string from the edge
        routes = {**self.valid_routes(held=0), ("POST", "/v1/chat/completions"): lambda: next(chats, (503, "x", None))}
        code, lines = self.phase_a(routes, [mstate(("ta-1", "RUNNING"))])
        self.assertEqual(code, 6)
        self.assertEqual(lines[-1]["violation"], "the open request did not get 503 mdf_tripped from the proxy")

    def test_passes_with_the_trip_the_zero_and_no_restart(self):
        chats = iter([(200, {"choices": []}, "forwarded"), (503, {"error": "mdf_tripped"}, "error")])
        routes = {**self.valid_routes(held=0), ("POST", "/v1/chat/completions"): lambda: next(chats)}
        states = [mstate(("ta-1", "RUNNING")), mstate(("ta-1", "RUNNING")), mstate()]
        code, lines = self.phase_a(routes, states)
        self.assertEqual(code, 0, lines[-1])
        self.assertEqual((lines[-1]["pass"], lines[-1]["restarts"]), (True, 0))
        self.assertIn("zero_since_hang_s", lines[-1])


# The start of vLLM's /metrics: the process collectors of prometheus_client come before vLLM's lines.
PROMETHEUS_HEAD = "".join(
    "# HELP python_gc_objects_collected_total Objects collected during gc\n"
    "# TYPE python_gc_objects_collected_total counter\n"
    'python_gc_objects_collected_total{generation="%d"} 1234.0\n' % i for i in range(60))


class TestConcurrency(Tasks):
    """c7.15 (20/11, `concurrency` on Modal: vLLM logged «Running: 32 reqs» while every /metrics read
    gave 0, because `Inference.get` keeps only the first 200 characters of a text body): the metric
    comes from the whole text, each reading is in the result, the body never is."""

    def concurrency(self, tail):
        page = self.dir / "page.png"
        page.write_bytes(b"png")
        served, posts = threading.Event(), []

        def metrics():
            served.set()
            return (200, PROMETHEUS_HEAD + tail, "forwarded")

        def chat():
            posts.append(1)
            if len(posts) == 1:
                self.clock.t += 1000.0              # the warm-up page: the busy loop's fake seconds stay under 4x
            else:
                served.wait(5.0)                    # the pages last until a reading of /metrics
            return (200, {"choices": []}, "forwarded")
        routes = {**self.valid_routes(held=0), ("GET", "/metrics"): metrics, ("POST", "/v1/chat/completions"): chat}
        t = self.tasks(["concurrency", "--input", str(page), "--n", "4", "--probe-while-busy"],
                       [mstate(("ta-1", "RUNNING"))], routes)
        return self.run_quiet(t.concurrency)

    def test_the_metric_past_the_start_of_the_body(self):
        self.assertGreater(len(PROMETHEUS_HEAD), 4096)
        code, lines = self.concurrency('vllm:num_requests_running{engine="0",model_name="m"} 30.0\n'
                                       'vllm:num_requests_running{engine="1",model_name="m"} 2.0\n'
                                       'vllm:num_requests_waiting{engine="0",model_name="m"} 0.0\n')
        res = lines[-1]
        self.assertEqual(code, 0, res)
        self.assertEqual((res["pass"], res["peak"]["running"], res["statuses"]), (True, 32.0, [200] * 4))
        self.assertGreaterEqual(res["metrics"]["with_running"], 1)
        self.assertEqual((res["metrics"]["last"][0]["status"], res["metrics"]["last"][0]["running"]), (200, 32.0))
        self.assertNotIn("python_gc", json.dumps(lines))       # the body is never in the output

    def test_metrics_read_apart_from_the_probes_see_the_real_peak(self):
        """MFL033 N6 (21/11, `concurrency` on RunPod, the same loop: each round waited first for the probe
        of /v1/models, up to 5 s, and /metrics missed the peak). The probes during the load answer only
        once it is over, the pages only once /metrics has seen them all running."""
        page = self.dir / "page.png"
        page.write_bytes(b"png")
        n, lock, st = 8, threading.Lock(), {"pages": 0, "inflight": 0}
        all_in, done = threading.Event(), threading.Event()

        def metrics():
            with lock:
                running = st["inflight"]
            if running >= n:
                all_in.set()
            return (200, PROMETHEUS_HEAD + 'vllm:num_requests_running{engine="0"} %d.0\n' % running, "forwarded")

        def chat():
            with lock:
                st["pages"] += 1
                load = st["pages"] > 1
                st["inflight"] += load
            if not load:
                self.clock.t += 1000.0              # the warm-up page
                return (200, {"choices": []}, "forwarded")
            all_in.wait(10.0)
            with lock:
                st["inflight"] -= 1
                if st["inflight"] == 0:
                    done.set()
            return (200, {"choices": []}, "forwarded")
        ready = self.models(0)

        def models():
            if st["pages"]:
                done.wait(10.0)                     # a probe held by the busy server
                return (200, {"data": [{"id": dv.MODEL}]}, "forwarded")
            return ready()
        routes = {**self.valid_routes(held=0), ("GET", "/metrics"): metrics, ("POST", "/v1/chat/completions"): chat,
                  ("GET", "/v1/models"): models}
        t = self.tasks(["concurrency", "--input", str(page), "--n", str(n), "--probe-while-busy"],
                       [mstate(("ta-1", "RUNNING"))], routes)
        code, lines = self.run_quiet(t.concurrency)
        res = lines[-1]
        self.assertTrue(all_in.is_set())
        self.assertEqual((res["peak"]["running"], res["statuses"]), (n, [200] * n))
        self.assertGreaterEqual(res["metrics"]["with_running"], 1)

    def test_no_metric_in_the_text_is_said(self):
        code, lines = self.concurrency("")
        res = lines[-1]
        self.assertEqual((code, res["pass"], res["peak"]["running"]), (6, False, 0))
        self.assertGreaterEqual(res["metrics"]["reads"], 1)
        self.assertEqual(res["metrics"]["with_running"], 0)
        self.assertEqual(res["metrics"]["last"][0]["status"], 200)

    def test_vllm_metrics(self):
        text = '# HELP vllm:num_requests_running x\nvllm:num_requests_running{a="1"} 3\nvllm:num_requests_running 1\n' \
               'vllm:num_requests_running_total 9\nvllm:num_requests_waiting{a="1"} 2.5\nvllm:num_requests_waiting bad\n'
        self.assertEqual(dv.vllm_metrics(text), {"running": 4.0, "waiting": 2.5})
        self.assertEqual(dv.vllm_metrics(PROMETHEUS_HEAD), {})


class TestAbandon(Tasks):
    """c7.16 (20/11: the request `concurrency` abandoned was the short chat, answered in 0.14 s before
    the client's 5 s, so nothing was abandoned): the abandoned request is a page that vLLM runs before
    the load and whose client gives up while it runs; App. D on it (r3 §5.3): one container, the same,
    after the load, and the other pages 200. The result keeps the chats of the log by
    source and Modal's inputs. Behind the fake edge a fake vLLM: /metrics counts the pages it runs."""

    N = 24

    def run_abandon(self, states, answer_abandoned=False, inputs=None, log_text=""):
        """`log_text` goes to the log of the session at the warm-up page, after readiness (the fast
        stop of the readiness wait reads the log)."""
        page = self.dir / "page.png"
        page.write_bytes(b"png")
        lock, arrived, all_in = threading.Lock(), threading.Event(), threading.Event()
        vllm, waited, self.posts = {"abandoned": 0, "load": 0}, [], []

        def handle(request):
            if (request.headers.get("modal-key"), request.headers.get("modal-secret")) != (TOKEN_ID, TOKEN_SECRET):
                return httpx.Response(401, text="modal-http: invalid credentials for proxy authorization")
            fwd, path = {"x-mdf-proxy": "forwarded"}, request.url.path
            if path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": dv.MODEL}]}, headers=fwd)
            if path == "/metrics":
                if not waited:                          # the abandoned request reaches vLLM before any page
                    waited.append(arrived.wait(5.0))
                with lock:
                    running = vllm["abandoned"] + vllm["load"]
                    if vllm["load"] >= self.N:
                        all_in.set()
                return httpx.Response(200, text=PROMETHEUS_HEAD + 'vllm:num_requests_running{engine="0"} %d.0\n'
                                      % running, headers=fwd)
            content = json.loads(request.content)["messages"][0]["content"]
            read = request.extensions["timeout"]["read"]
            with lock:
                self.posts.append(("page" if isinstance(content, list) else "chat", read))
                first = len(self.posts) == 1
            if first:                                   # the warm-up page
                self.clock.t += 1000.0
                with open(self.log, "a") as fh:
                    fh.write(log_text)
            elif read == 5.0:                           # the abandoned request: vLLM runs it, its client gives up
                with lock:
                    vllm["abandoned"] += 1
                arrived.set()
                if not answer_abandoned:
                    raise httpx.ReadTimeout("the client gave up")
            else:                                       # a page of the load: until /metrics saw them all
                with lock:
                    vllm["load"] += 1
                all_in.wait(5.0)
                with lock:
                    vllm["load"] -= 1
            return httpx.Response(200, json={"choices": []}, headers=fwd)

        transport = httpx.MockTransport(handle)
        t = self.tasks(["concurrency", "--input", str(page), "--n", str(self.N), "--abandon", "1", "--probe-while-busy"],
                       states, {}, inputs=inputs)
        t.transport = transport
        real = httpx.Client

        def client(*args, **kw):                        # never the network: c7.15 sent it without the transport
            kw["transport"] = kw.get("transport") or transport
            return real(*args, **kw)
        with mock.patch.object(httpx, "Client", client):
            return self.run_quiet(t.concurrency)

    def test_the_abandoned_request_is_a_page_run_before_the_load_and_given_up(self):
        code, lines = self.run_abandon([mstate(("ta-1", "RUNNING"))])
        res = lines[-1]
        self.assertEqual(code, 0, res)
        self.assertEqual(self.posts[:2], [("page", 140.0), ("page", 5.0)])      # the warm-up, then the abandoned page
        self.assertEqual((len(self.posts), set(self.posts[2:])), (2 + self.N, {("page", 140.0)}))
        ab = res["abandon"]
        self.assertEqual((ab["n"], ab["after_s"], ab["exercised"], ab["running_before_load"]), (1, 5.0, True, 1))
        self.assertEqual(ab["requests"][0]["error"], "ReadTimeout")
        self.assertIn("t_send", ab["requests"][0])
        self.assertEqual((res["pass"], res["violation"], res["containers"]), (True, None, ["ta-1"]))
        self.assertEqual(res["statuses"], [200] * self.N)
        self.assertGreaterEqual(res["peak"]["running"], 24)
        self.assertEqual(len(res["after_load"]), 3)                            # 30 s after the load, every 10 s
        self.assertGreaterEqual(res["readings"], 4)                            # readiness, then 3 after the load

    def test_the_container_lives_through_the_abandoned_page(self):
        up = mstate(("ta-1", "RUNNING"))
        cases = (("replaced", [up, up, mstate(("ta-2", "RUNNING"))], "a restart: worker ta-2 after ['ta-1']"),
                 ("gone", [up, up, mstate()], "no container at"),
                 ("two", [up, mstate(("ta-1", "RUNNING"), ("ta-2", "RUNNING"))], "total 2 > 1"))
        for name, states, said in cases:
            with self.subTest(name):
                code, lines = self.run_abandon(states)
                res = lines[-1]
                self.assertEqual((code, res["pass"]), (6, False), res)
                self.assertIn(said, res["violation"])

    def test_a_page_answered_before_its_client_gave_up_was_not_abandoned(self):
        code, lines = self.run_abandon([mstate(("ta-1", "RUNNING"))], answer_abandoned=True)
        res = lines[-1]
        self.assertEqual((code, res["pass"], res["abandon"]["exercised"]), (6, False, False), res)
        self.assertEqual(res["abandon"]["requests"][0]["status"], 200)
        self.assertTrue(res["violation"].startswith("an abandoned page was not abandoned"), res["violation"])

    def test_the_chats_of_the_log_by_source_and_the_inputs_of_modal(self):
        def row(source, line):
            return "2026-11-20T09:29:10.048Z\t+30.0\tta-1\t%s\t2026-11-20 10:29:10+01:00\t%s\n" % (source, line)
        chat = {"event": "mdf_request", "kind": "forwarded", "method": "POST", "path": "/v1/chat/completions",
                "status": 200, "t": 1795166950.0}
        access = '(APIServer pid=9) INFO:     2.231.198.124:0 - "POST /v1/chat/completions HTTP/1.1" 200 OK'
        log_text = "".join([
            row("container", json.dumps(chat)), row("container", json.dumps(dict(chat, t=1795166951.0))),
            row("container", json.dumps(dict(chat, kind="error", status=502, code="mdf_upstream"))),
            row("container", json.dumps(dict(chat, method="GET", path="/metrics"))),
            row("container", json.dumps({"event": "mdf_trip", "rule": "W3"})),
            row("container", access), row("container", access),
            row("system", "   POST /v1/chat/completions -> 200 OK  (duration: 6.59 s, execution: 6.57 s)"),
            row("system", "   POST /v1/chat/completions -> client went away  (duration: 5.01 s)"),   # a made-up result
            row("system", "   GET /v1/models -> 502 Bad Gateway  (duration: 40.2 ms, execution: 13.9 ms)"),
            row("session", json.dumps(chat)), "a line cut sho"])
        inputs = [{"enqueued_at": "a", "started_at": "a1", "container_id": "ta-1", "execution_time_seconds": 6.6,
                   "status": "success", "function_call_id": "fc-1"},
                  {"enqueued_at": "b", "started_at": "b1", "container_id": "ta-1", "execution_time_seconds": 11.9,
                   "status": "success"},
                  {"enqueued_at": "c", "started_at": "c1", "container_id": "ta-1", "execution_time_seconds": 5.0,
                   "status": "terminated"},
                  {"enqueued_at": "d", "started_at": None, "container_id": None, "execution_time_seconds": None,
                   "status": "pending"}]
        code, lines = self.run_abandon([mstate(("ta-1", "RUNNING"))], inputs=inputs, log_text=log_text)
        res = lines[-1]
        self.assertEqual(code, 0, res)
        ab = res["abandon"]
        self.assertEqual(ab["log"], {
            "sent": 2 + self.N, "proxy": {"forwarded 200": 2, "error 502 mdf_upstream": 1}, "proxy_total": 3,
            "vllm": {"200": 2}, "modal": {"200 OK": 1, "client went away": 1}, "modal_total": 2,
            "modal_lines": [[30.0, "200 OK", "duration: 6.59 s, execution: 6.57 s"],
                            [30.0, "client went away", "duration: 5.01 s"]],
            "trips": [{"rule": "W3", "since_arm": 30.0}]})

        def brief(i):
            return {k: inputs[i][k] for k in ("enqueued_at", "started_at", "execution_time_seconds", "status")}
        self.assertEqual(ab["inputs"], {"since": round(ab["requests"][0]["t_send"] - 1, 3), "n": 4,
                                        "status": {"success": 2, "terminated": 1, "pending": 1}, "containers": ["ta-1"],
                                        "not_success": [brief(2), brief(3)],
                                        "longest": [brief(1), brief(0), brief(2), brief(3)]})
        self.assertEqual(self.modal.input_since, [ab["inputs"]["since"]])

    def test_a_failed_reading_of_the_inputs_is_said_not_a_criterion(self):
        code, lines = self.run_abandon([mstate(("ta-1", "RUNNING"))],
                                       inputs=pv.ProviderError("modal function calls mdf-surya/serve: exit 1 boom"))
        res = lines[-1]
        self.assertEqual(code, 0, res)
        self.assertEqual(res["abandon"]["inputs"]["error"], "modal function calls mdf-surya/serve: exit 1 boom")


class NotHandled(Exception):
    """The signal reached the test's own handler: the driver did not handle it."""


def fallback_handler(test, sig):
    """A handler of the test for `sig` while it runs: without it a driver that does not handle the
    signal would end the whole test run."""
    import signal

    def handler(signum, frame):
        raise NotHandled(signal.Signals(signum).name)
    old = signal.signal(sig, handler)
    test.addCleanup(signal.signal, sig, old)
    return handler


class TestEdgeRefusal(Tasks):
    """c7.13 (19/11: Modal's edge answered every readiness GET of `probe-valid` at once, for 765 s,
    and the driver kept them in memory): a 401, 403 or 404 without `X-MDF-Proxy` stops the wait at the
    first one, with the status, the names of the headers and the start of the body, never a key."""

    def refusing(self, status):
        def handle(request):
            self.calls.append((request.method, request.url.path))
            echo = "modal-http: invalid credentials %s %s" % (request.headers.get("modal-key"),
                                                             request.headers.get("modal-secret"))
            return httpx.Response(status, text=echo, headers={"content-type": "text/plain; charset=utf-8",
                                                              "x-edge-value": "header-value-xyz"})
        return httpx.MockTransport(handle)

    def test_a_refusal_of_the_edge_stops_the_wait_at_once(self):
        for status in (401, 403, 404):
            for argv in (["probe-valid"], ["concurrency", "--input", "page.png"], ["fault-test", "--phase", "A"]):
                with self.subTest(status=status, driver=argv[0]):
                    self.calls = []
                    t = self.tasks(argv, [mstate()], {})
                    t.transport = self.refusing(status)
                    fn = {"probe-valid": t.probe_valid, "concurrency": t.concurrency, "fault-test": t.fault_test}[argv[0]]
                    out = io.StringIO()
                    with redirect_stdout(out):
                        code = fn()
                    lines = [json.loads(x) for x in out.getvalue().splitlines()]
                    self.assertEqual(code, 6)
                    self.assertEqual(self.calls, [("GET", "/v1/models")])            # one GET, no retry, no POST
                    res = lines[-1]
                    ready = res["probe_valid"]["ready"] if argv[0] == "probe-valid" else res["ready"]
                    self.assertEqual((ready["ready"], ready["stopped"], ready["tries"]),
                                     (False, "refused at the edge: HTTP %d" % status, 1))
                    edge = ready["edge"]
                    self.assertEqual((edge["status"], edge["content_type"]), (status, "text/plain; charset=utf-8"))
                    self.assertIn("x-edge-value", edge["header_names"])
                    self.assertEqual(edge["body_start"], "modal-http: invalid credentials [key] [key]")
                    tries = [x for x in lines if "ready_try" in x]
                    self.assertEqual([(x["ready_try"], x["status"], x["edge"]["status"]) for x in tries], [(1, status, status)])
                    for secret in (TOKEN_ID, TOKEN_SECRET, "header-value-xyz"):
                        self.assertNotIn(secret, out.getvalue())

    def test_an_answer_of_the_proxy_or_a_5xx_is_not_a_refusal(self):
        answers = iter([(404, {}, "error"), (401, {}, "error"), (503, {}, None), (200, {"data": [{"id": dv.MODEL}]}, "forwarded")])
        routes = {**self.valid_routes(), ("GET", "/v1/models"): lambda: next(answers)}
        code, lines = self.run_quiet(self.tasks(["probe-valid"], [mstate(("ta-1", "RUNNING"))], routes).probe_valid)
        self.assertEqual(code, 0, lines[-1])
        self.assertEqual(lines[-1]["probe_valid"]["ready"]["tries"], 4)

    def test_each_attempt_is_a_line_as_soon_as_it_happens(self):
        out, seen = io.StringIO(), []
        models = self.models(held=3)

        def answer():
            seen.append(sum(1 for x in out.getvalue().splitlines() if '"ready_try"' in x))   # lines before this GET
            return models()
        routes = {**self.valid_routes(), ("GET", "/v1/models"): answer}
        t = self.tasks(["probe-valid"], [mstate(("ta-1", "RUNNING"))], routes)
        with redirect_stdout(out):
            code = t.probe_valid()
        self.assertEqual(code, 0)
        self.assertEqual(seen, [0, 1, 2, 3])
        tries = [json.loads(x) for x in out.getvalue().splitlines() if '"ready_try"' in x]
        self.assertEqual([(x["ready_try"], x["status"], x["error"], x["s"]) for x in tries],
                         [(1, None, "ReadTimeout", 30.0), (2, None, "ReadTimeout", 65.0), (3, None, "ReadTimeout", 100.0),
                          (4, 200, None, 105.0)])

    def test_sigterm_or_sigint_prints_the_result_so_far(self):
        import os
        import signal
        for sig in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=sig.name):
                calls = []

                def answer():
                    calls.append(1)
                    if len(calls) == 3:
                        os.kill(os.getpid(), sig)                       # the session stops the driver
                        time.sleep(5)
                    self.clock.t += 30.0
                    return httpx.ReadTimeout("held by Modal")
                argv = ["--env-file", str(self.env), "--server", "modal-l4", "--servers", str(self.servers), "probe-valid",
                        "--app", "mdf-surya", "--spend-ledger", str(self.ledger)]
                before = fallback_handler(self, sig)          # a driver without its own handler fails, not pytest
                out = io.StringIO()
                with redirect_stdout(out):
                    code = mt.main(argv, modal=StateOnly([mstate(("ta-1", "RUNNING"))]),
                                   transport=self.edge({("GET", "/v1/models"): answer}), clock=self.clock,
                                   sleep=self.clock.sleep, environ={serverspend.SESSION_ENV: self.arm_id,
                                                                    wl.LOG_ENV: str(self.log)})
                lines = [json.loads(x) for x in out.getvalue().splitlines()]
                self.assertEqual(code, 130)
                self.assertEqual([x["ready_try"] for x in lines if "ready_try" in x], [1, 2])
                last = lines[-1]
                self.assertEqual(last["stopped_by"], sig.name)
                self.assertEqual(last["partial"]["ready_wait"]["tries"], 2)
                self.assertEqual(last["partial"]["workers"], ["ta-1"])
                self.assertIs(signal.getsignal(sig), before)               # the previous handler is back


WAITING_TEXT = ("Function 'serve' (fu-sF4Wg5rE4HlmgcKfKPvp5Q) is waiting to be scheduled on a GPU_L4 worker. "
                "We are actively working on acquiring more capacity for your workload.")


def cli_time(t):
    """A time as `modal app logs --timestamps` printed it on 20/11: whole seconds, local offset +01:00."""
    from datetime import datetime, timedelta, timezone
    return datetime.fromtimestamp(int(t), timezone(timedelta(hours=1))).isoformat(sep=" ")


class Timed(StateOnly):
    """The app with no container until `container_at` (clock time), then one PENDING."""

    def __init__(self, clock, container_at=None, system=None):
        super().__init__([mstate()], system)
        self.clock, self.container_at = clock, container_at

    def state(self):
        if self.container_at is not None and self.clock.t >= self.container_at:
            return mstate(("ta-1", "PENDING"))
        return mstate()


class TestNoCapacity(Tasks):
    """c7.14 (20/11: no L4 free on Modal; its scheduler wrote "waiting to be scheduled on a GPU_L4
    worker" at +110 s from the arm, no container came in 13 minutes, the stream `-f` of the follower
    never delivered the line, and the wait would have gone on to 1260 s): the system lines are fetched
    without `-f` at most every 30 s, and that line followed by no container for 300 s stops the wait
    with `modal_no_capacity`, exit 7, its text and the GPU it names; no GET runs past the deadline."""

    def setUp(self):
        super().setUp()
        self.t_arm = self.clock.t - 20                       # the arm row of Tasks.setUp
        self.gets, self.fetch_times = [], []

    def waiting(self, at=110.0, fail=False):
        """The system lines of the app: the scheduler's line from `at` s after the arm."""
        def fetch(since):
            self.fetch_times.append(self.clock.t)
            if fail:
                raise pv.ProviderError("modal app logs: exit 1 Error Token not found")
            if self.clock.t < self.t_arm + at:
                return []
            return ["%s fu-sF4Wg5rE4HlmgcKfKPvp5Q %s" % (cli_time(self.t_arm + at), WAITING_TEXT)]
        return fetch

    def queued(self, ready_at=None):
        """Modal's edge while the input waits in the queue: a readiness GET lasts its own read timeout,
        until `ready_at` (clock time); the other paths answer as `valid_routes`."""
        routes = self.valid_routes(held=0)

        def handle(request):
            self.calls.append((request.method, request.url.path))
            if request.url.path == "/v1/models":
                read = request.extensions["timeout"]["read"]
                self.gets.append((round(self.clock.t, 3), read))
                if ready_at is None or self.clock.t < ready_at:
                    self.clock.t += read
                    raise httpx.ReadTimeout("queued by Modal", request=request)
                return httpx.Response(200, json={"data": [{"id": dv.MODEL}]}, headers={"x-mdf-proxy": "forwarded"})
            status, body, proxy = routes.get((request.method, request.url.path), (404, {}, None))
            return httpx.Response(status, json=body, headers={"x-mdf-proxy": proxy} if proxy else {})
        return httpx.MockTransport(handle)

    def driver(self, argv, modal, ready_at=None):
        t = self.tasks(argv, [mstate()], {})
        t.modal, t.transport = modal, self.queued(ready_at)
        self.modal = modal
        fn = {"probe-valid": t.probe_valid, "concurrency": t.concurrency, "fault-test": t.fault_test}[argv[0]]
        return self.run_quiet(fn)

    def test_waiting_without_a_container_stops_after_300_s(self):
        code, lines = self.driver(["probe-valid"], Timed(self.clock, system=self.waiting()))
        self.assertEqual(code, mt.NO_CAPACITY_EXIT, lines[-1])
        ready = lines[-1]["probe_valid"]["ready"]
        self.assertEqual((ready["ready"], ready["code"]), (False, "modal_no_capacity"))
        self.assertEqual(ready["stopped"], 'modal_no_capacity: no container 300 s after "%s" (requested L4/A10G)'
                         % WAITING_TEXT)
        waiting = ready["scheduler"]["waiting"]
        self.assertEqual((waiting["text"], waiting["gpu"], waiting["t_from"]), (WAITING_TEXT, "GPU_L4", self.t_arm + 110))
        self.assertEqual(ready["scheduler"]["requested"], ["L4", "A10G"])
        deadline = self.t_arm + 110 + 300
        self.assertGreaterEqual(ready["since_arm_s"], 410.0)                    # never before the 300 s
        self.assertLessEqual(ready["since_arm_s"], 410.0 + 5 + 1)               # the pause after the last GET
        for t, read in self.gets:
            self.assertLessEqual(t + read, deadline + 1e-6)                     # no GET runs past the deadline
        self.assertNotIn(("POST", "/v1/chat/completions"), self.calls)
        self.assertEqual(len([x for x in lines if "modal_system" in x]), 1)     # the line once, as soon as read
        self.assertEqual(self.modal.fetches, [self.t_arm] * len(self.modal.fetches))   # since the arm
        gaps = [b - a for a, b in zip(self.fetch_times, self.fetch_times[1:])]
        self.assertTrue(gaps and min(gaps) >= mt.READY_READ_EVERY_S, gaps)       # at most every 30 s
        self.assertEqual(ready["workers"], [])

    def test_every_readiness_wait_of_modal(self):
        for argv in (["concurrency", "--input", "page.png"], ["fault-test", "--phase", "A"],
                     ["fault-test", "--phase", "B"]):
            with self.subTest(driver=" ".join(argv)):
                self.clock.t, self.calls = self.t_arm + 20, []
                code, lines = self.driver(argv, Timed(self.clock, system=self.waiting()))
                self.assertEqual(code, mt.NO_CAPACITY_EXIT, lines[-1])
                self.assertEqual(lines[-1]["ready"]["code"], "modal_no_capacity")
                self.assertNotIn(("POST", "/v1/chat/completions"), self.calls)

    def test_a_container_lifts_the_stop(self):
        """A container listed before the deadline, or only by the reading made once it has passed: the
        capacity came, the start goes on under READY_WITHIN_S."""
        for name, container_at in (("before", 300.0), ("at the reading after the deadline", 412.0)):
            with self.subTest(name):
                self.clock.t = self.t_arm + 20
                self.gets, self.calls = [], []
                modal = Timed(self.clock, container_at=self.t_arm + container_at, system=self.waiting())
                code, lines = self.driver(["probe-valid"], modal, ready_at=self.t_arm + 700)
                self.assertEqual(code, 0, lines[-1])
                ready = lines[-1]["probe_valid"]["ready"]
                self.assertEqual((ready["ready"], ready.get("code")), (True, None))
                self.assertEqual(ready["scheduler"]["waiting"]["gpu"], "GPU_L4")
                self.assertEqual(ready["workers"], ["ta-1"])

    def test_without_the_line_or_with_a_failed_fetch_the_wait_goes_on(self):
        code, lines = self.driver(["probe-valid"], Timed(self.clock))
        ready = lines[-1]["probe_valid"]["ready"]
        self.assertEqual((code, ready["ready"], ready.get("code"), ready.get("stopped")), (6, False, None, None))
        self.assertGreaterEqual(ready["seconds"], mt.READY_WITHIN_S)            # the limit of before, unchanged
        self.assertIsNone(ready["scheduler"]["waiting"])
        self.clock.t = self.t_arm + 20
        modal = Timed(self.clock, container_at=self.t_arm, system=self.waiting(fail=True))
        code, lines = self.driver(["probe-valid"], modal, ready_at=self.t_arm + 200)
        ready = lines[-1]["probe_valid"]["ready"]
        self.assertEqual((code, ready["ready"]), (0, True), lines[-1])
        self.assertIn("Token not found", ready["scheduler"]["errors"][0])       # recorded, not a criterion

    def test_a_signal_prints_the_scheduler_so_far(self):
        import os
        import signal
        modal = Timed(self.clock, system=self.waiting(at=40.0))

        def stop_soon(request):
            self.clock.t += 30.0
            if self.clock.t > self.t_arm + 150:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
            raise httpx.ReadTimeout("queued by Modal", request=request)
        argv = ["--env-file", str(self.env), "--server", "modal-l4", "--servers", str(self.servers), "probe-valid",
                "--app", "mdf-surya", "--spend-ledger", str(self.ledger)]
        fallback_handler(self, signal.SIGTERM)
        out = io.StringIO()
        with redirect_stdout(out):
            code = mt.main(argv, modal=modal, transport=httpx.MockTransport(stop_soon), clock=self.clock,
                           sleep=self.clock.sleep, environ={serverspend.SESSION_ENV: self.arm_id, wl.LOG_ENV: str(self.log)})
        last = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual((code, last["stopped_by"]), (130, "SIGTERM"))
        self.assertEqual(last["partial"]["scheduler"]["waiting"]["gpu"], "GPU_L4")


class TestGpu(Tasks):
    """c7.14: the readiness names the GPU the proxy found at its start (`mdf_gpu` in the log)."""

    def test_the_gpu_of_the_proxy_in_the_readiness(self):
        self.write_log('{"event": "mdf_start", "provider": "modal"}',
                       '{"event": "mdf_gpu", "names": ["NVIDIA A10G"], "source": "nvidia-smi", "error": null}')
        code, lines = self.run_quiet(self.tasks(["probe-valid"], [mstate(("ta-1", "RUNNING"))],
                                                self.valid_routes(held=1)).probe_valid)
        self.assertEqual(code, 0, lines[-1])
        ready = lines[-1]["probe_valid"]["ready"]
        self.assertEqual(ready["gpu"], ["NVIDIA A10G"])
        self.assertEqual(ready["proxy_events"]["mdf_gpu"]["source"], "nvidia-smi")


CA, CB, CC = "ta-01M7W785K06WPNX3AE5K120CNR", "ta-01M7W78NW2BKMWWTYTAYES6SYR", "ta-01M7W79QQ3CKMWWTYTAYES6ZZZ"


def listed(s, cid, status="PENDING", start=None):
    """A line of the follower's listing (`ModalFollower.list_once`) at `s` s since the arm."""
    ev = {"event": "worker", "status": status}
    if status != "GONE":
        ev["start"] = start
    return (s, cid, "session", json.dumps(ev))


def proxy(s, cid, event, **fields):
    """A JSON line of a container's proxy at `s` s since the arm (its `t` filled in by `LogScript`)."""
    return (s, cid, "container", dict(fields, event=event))


class LogScript:
    """The log of the session as the follower writes it: (s since the arm, container, source, line)
    rows, appended to the file once the fake clock reaches them (`flush`, at each request of the fake
    edge and each reading of the fake app). In one listing the GONE lines come first, as in
    `ModalFollower.list_once`."""

    def __init__(self, path, clock, t_arm, rows):
        self.path, self.clock, self.t_arm = path, clock, t_arm
        self.rows = sorted(rows, key=lambda r: (r[0], 0 if '"GONE"' in str(r[3]) else 1))
        follow = json.dumps({"event": "follow", "provider": "modal", "t_arm": wl.iso(t_arm, ms=True)})
        self.rows.insert(0, (-16.0, "-", "session", follow))

    def flush(self):
        now = self.clock.t - self.t_arm
        with open(self.path, "a") as fh:
            while self.rows and self.rows[0][0] <= now:
                s, cid, source, line = self.rows.pop(0)
                t = self.t_arm + s
                if isinstance(line, dict):
                    line = json.dumps(dict(line, t=round(t, 3)))
                fh.write("%s\t+%.1f\t%s\t%s\t%s\t%s\n" % (wl.iso(t, ms=True), s, cid, source,
                                                          "" if source == "session" else cli_time(t), line))


class Scripted(StateOnly):
    """The app as Modal lists it at the fake clock: `spans` of (container, from s, to s) since the arm;
    each reading first writes the lines of the log due by then. The follower's lines for the spans
    (PENDING at the start, GONE at the end) are made by `rows`."""

    def __init__(self, clock, t_arm, spans, script):
        super().__init__([mstate()])
        self.clock, self.t_arm, self.spans, self.script = clock, t_arm, spans, script
        self.read_at = []

    @staticmethod
    def rows(spans):
        out = []
        for cid, a, b in spans:
            out.append(listed(a, cid))
            if b is not None:
                out.append(listed(b, cid, "GONE"))
        return out

    def state(self):
        self.script.flush()
        s = self.clock.t - self.t_arm
        self.read_at.append(round(s, 1))
        return mstate(*[(cid, "RUNNING") for cid, a, b in self.spans if a <= s and (b is None or s < b)])


def starting(cid, at, gpu="NVIDIA A10G"):
    """The first lines of a container's proxy: start, GPU, weights, vLLM started, a 502 of the proxy."""
    return [proxy(at, cid, "mdf_start", provider="modal"), proxy(at, cid, "mdf_gpu", names=[gpu], source="nvidia-smi"),
            proxy(at + 1.8, cid, "mdf_weights_check", ok=True, phase="start"), proxy(at + 1.8, cid, "mdf_vllm_start"),
            proxy(at + 1.9, cid, "mdf_request", kind="error", method="GET", path="/v1/models", status=502,
                  code="mdf_upstream")]


def answering(cid, at):
    """vLLM answers through the proxy of `cid` from `at` s: first /health, first /v1/models, ready."""
    return [proxy(at, cid, "mdf_vllm_first_health", status=200), proxy(at, cid, "mdf_vllm_first_models", status=200),
            proxy(at, cid, "mdf_vllm_models_200"), proxy(at, cid, "mdf_ready")]


class TestRuleA(Tasks):
    """c7.17 (20/11, `concurrency-2`: the first container lived 11 s during the readiness, between two
    readings of the driver, which come every 30 s, and a second one served; the follower listed both
    every 5 s). Every wait and every phase reads the containers of the log of the session too, and
    rule (a) of Davide (20/11 11:31) judges them: a replacement before the first answer of vLLM, without
    overlap and without inferences, is declared, not a stop; an overlap, or a replacement after
    readiness, is a violation and the fast stop."""

    def setUp(self):
        super().setUp()
        self.t_arm = self.clock.t - 20                       # the arm row of Tasks.setUp

    def run_driver(self, argv, spans, lines, ready_at, chats=None, log=True):
        """A driver of `argv` on the app of `spans` with the log of `lines` (plus the follower's lines of
        the spans); /v1/models is 502 from the proxy until `ready_at` s since the arm, then 200."""
        script = LogScript(self.log, self.clock, self.t_arm, Scripted.rows(spans) + list(lines))
        modal = Scripted(self.clock, self.t_arm, spans, script)
        routes = self.valid_routes(held=0)

        def models():
            script.flush()
            if self.clock.t - self.t_arm < ready_at:
                return (502, {"error": "mdf_upstream"}, "error")
            return (200, {"data": [{"id": dv.MODEL}]}, "forwarded")
        routes[("GET", "/v1/models")] = models
        if chats is not None:
            routes[("POST", "/v1/chat/completions")] = lambda: next(chats)
        t = self.tasks(argv, [mstate()], routes, log=log)
        t.modal = self.modal = modal
        fn = {"probe-valid": t.probe_valid, "fault-test": t.fault_test}[argv[0]]
        with mock.patch.object(dv.Inference, "abandon", lambda self, after_s: None):     # never the network
            code, lines = self.run_quiet(fn)
        self.lines = lines
        return code, lines[-1]

    def test_a_replacement_before_vllm_answered_is_declared_not_a_stop(self):
        """The readings saw the first container (+50 s) and then the second (+80 s): before c7.17 a
        restart, and the fast stop; with rule (a), a fact declared at once and in the result."""
        spans = [(CA, 25.0, 55.0), (CB, 62.0, None)]
        lines = starting(CA, 25.5) + starting(CB, 61.0) + answering(CB, 290.0)
        code, res = self.run_driver(["probe-valid"], spans, lines, ready_at=290.0)
        self.assertEqual(code, 0, res)
        ready = res["probe_valid"]["ready"]
        self.assertEqual((ready["ready"], ready["workers"], ready["served"]), (True, [CA, CB], wl.short(CB)))
        self.assertEqual(len(ready["replacements"]), 1)
        row = res["replacements"][0]
        self.assertEqual((row["old"]["id"], row["new"]["id"]), ("ta-01M7W785K…", "ta-01M7W78NW…"))
        self.assertEqual((row["old"]["seen_s"], row["old"]["gone_s"], row["new"]["seen_s"]), (25.0, 55.0, 61.0))
        self.assertEqual((row["old"]["gpu"], row["new"]["gpu"]), (["NVIDIA A10G"], ["NVIDIA A10G"]))
        self.assertEqual((row["gap_s"], row["old_vllm_s"], row["old_inference_lines"]), (6.0, None, 0))
        self.assertIn("seen_utc", row["old"])
        declared = [x for x in self.lines if "replacement" in x]
        self.assertEqual((len(declared), declared[0]["rule"], declared[0]["old"]["id"]), (1, "a", "ta-01M7W785K…"))
        self.assertEqual([c["id"] for c in res["containers_log"]["containers"]], [wl.short(CA), wl.short(CB)])
        self.log.write_text("")                              # without the log of the session: as before c7.17
        self.clock.t = self.t_arm + 20
        code, res = self.run_driver(["probe-valid"], spans, lines, ready_at=290.0, log=False)
        self.assertEqual(code, 6, res)
        self.assertIn("a restart: worker %s" % CB, res["probe_valid"]["ready"]["stopped"])

    def test_two_containers_at_once_stop_the_wait(self):
        """Between two readings (+50 s and +80 s) that each saw one container at most."""
        cases = (("in one listing", [(CA, 25.0, 45.0), (CB, 40.0, None)], starting(CA, 25.5) + starting(CB, 40.5),
                  "two containers at once: ta-01M7W785K… and ta-01M7W78NW… in the same listing at +40.0 s"),
                 ("by their lines", [(CA, 25.0, 39.0), (CB, 41.0, None)],
                  starting(CA, 25.5) + starting(CB, 36.0) + [proxy(38.5, CA, "mdf_request", kind="error", method="GET",
                                                                   path="/v1/models", status=502)],
                  "two containers at once: ta-01M7W785K… wrote 2.5 s after ta-01M7W78NW… wrote its first line"))
        for name, spans, lines, said in cases:
            with self.subTest(name):
                self.log.write_text("")
                self.clock.t = self.t_arm + 20
                code, res = self.run_driver(["probe-valid"], spans, lines, ready_at=290.0)
                ready = res["probe_valid"]["ready"]
                self.assertEqual((code, ready["ready"]), (6, False), res)
                self.assertEqual(ready["stopped"], said)
                self.assertLess(ready["seconds"], 60)          # at the first check after the evidence
                self.assertNotIn(("POST", "/v1/chat/completions"), self.calls)

    def test_a_replacement_after_readiness_or_after_vllm_answered(self):
        cases = (("after readiness, between two readings", [(CA, 25.0, 292.0), (CC, 293.0, 296.0)],
                  starting(CA, 25.5) + answering(CA, 285.0), 290.0,
                  "a container replaced after readiness: ta-01M7W785K… (+25.0 to +292.0 s, ['NVIDIA A10G']) "
                  "then ta-01M7W79QQ… (from +293.0 s, None)"),
                 ("before readiness, after the first answer of vLLM", [(CA, 25.0, 55.0), (CB, 62.0, None)],
                  starting(CA, 25.5) + [proxy(45.0, CA, "mdf_vllm_first_health", status=503)] + starting(CB, 61.0)
                  + answering(CB, 290.0), 290.0,
                  "a container replaced after the first answer of vLLM (+45.0 s, mdf_vllm_first_health)"),
                 ("before readiness, with an inference on it", [(CA, 25.0, 55.0), (CB, 62.0, None)],
                  starting(CA, 25.5) + [(46.0, CA, "system", "    POST /v1/chat/completions -> 502 Bad Gateway  "
                                                            "(duration: 1.0 s)")] + starting(CB, 61.0)
                  + answering(CB, 290.0), 290.0,
                  "a container replaced with an inference on it (1 lines)"))
        for name, spans, lines, ready_at, said in cases:
            with self.subTest(name):
                self.log.write_text("")
                self.clock.t, self.calls = self.t_arm + 20, []
                code, res = self.run_driver(["probe-valid"], spans, lines, ready_at=ready_at)
                self.assertEqual((code, res["pass"]), (6, False), res)
                said_where = res["violation"] if res["probe_valid"]["ready"]["ready"] else res["probe_valid"]["ready"]["stopped"]
                self.assertTrue(said_where.startswith(said), said_where)
                self.assertEqual(res["replacements"], [])

    def test_a_container_between_two_readings(self):
        """Seen by the log only, as `concurrency-2` (+10.8 to +22.3 s, between readings at +4.3 and +34
        s): before c7.17 nothing of it reached the result; now the replacement is declared, and a
        restart after the hang of phase A is the violation it is."""
        with self.subTest("the readiness of concurrency-2"):
            spans = [(CA, 25.0, 37.0), (CB, 43.0, None)]
            lines = starting(CA, 25.5) + starting(CB, 41.5) + answering(CB, 290.0)
            code, res = self.run_driver(["probe-valid"], spans, lines, ready_at=290.0)
            self.assertEqual(code, 0, res)
            self.assertEqual(res["probe_valid"]["ready"]["workers"], [CB])             # the readings saw one
            self.assertEqual([s for s in self.modal.read_at if 25 <= s < 37], [])    # none in its life
            self.assertEqual([(r["old"]["id"], r["old"]["seen_s"], r["old"]["gone_s"], r["new"]["seen_s"])
                              for r in res.get("replacements", [])], [("ta-01M7W785K…", 25.0, 37.0, 41.5)])
        with self.subTest("phase A, a restart after the hang"):
            self.log.write_text("")
            self.clock.t, self.calls = self.t_arm + 20, []
            chats = iter([(200, {"choices": []}, "forwarded"), (503, {"error": "mdf_tripped"}, "error")])
            spans = [(CA, 25.0, 115.0), (CC, 143.0, 146.0)]                # the zero, then 3 s between readings
            code, res = self.run_driver(["fault-test", "--phase", "A"], spans,
                                        starting(CA, 25.5) + answering(CA, 95.0), ready_at=100.0, chats=chats)
            self.assertEqual((code, res["pass"]), (6, False), res)
            self.assertTrue(res["violation"].startswith("a container restarted after the hang: ta-01M7W785K…"),
                            res["violation"])
            self.assertEqual(res["restarts"], 0)                             # the readings never saw it

    def test_phase_b_observes_the_restarts_after_the_fault(self):
        chats = [(200, {"choices": []}, "forwarded")] + [httpx.ReadTimeout("crashed")] * 2
        cases = (("a restart, observed", [(CA, 25.0, 110.0), (CB, 116.0, None)], 0, 1),
                 ("two at once after the fault", [(CA, 25.0, 115.0), (CB, 112.0, None)], 6, 0))
        for name, spans, expected, restarts in cases:
            with self.subTest(name):
                self.log.write_text("")
                self.clock.t, self.calls = self.t_arm + 20, []
                lines = starting(CA, 25.5) + answering(CA, 95.0) + [proxy(100.5, CA, "mdf_fault",
                                                                         fault="crash-after-first")]
                code, res = self.run_driver(["fault-test", "--phase", "B", "--observe-seconds", "60"], spans,
                                            lines + starting(CB, 116.5), ready_at=100.0, chats=iter(chats))
                self.assertEqual(code, expected, res)
                self.assertEqual(len(res["restarts_after_fault"]), restarts)
                if expected:
                    self.assertTrue(res["violation"].startswith("two containers at once"), res["violation"])
                else:
                    row = res["restarts_after_fault"][0]
                    self.assertEqual((row["old"]["gone_s"], row["new"]["seen_s"], row["new"]["gpu"]),
                                     (110.0, 116.0, ["NVIDIA A10G"]))
                    self.assertEqual(res["containers_log"]["faults"][0]["fault"], "crash-after-first")


class TestLives(unittest.TestCase):
    """c7.17: `worker_logs.Lives` reads the containers of a Modal window from the file of the session."""

    def test_the_containers_of_a_window(self):
        clock, t_arm = Clock(), 1_800_000_000.0
        clock.t = t_arm + 1000
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        path = tmp / "worker-log.partial"
        access = '(APIServer pid=9) INFO:     2.231.198.124:0 - "POST /v1/chat/completions HTTP/1.1" 200 OK'
        LogScript(path, clock, t_arm, starting(CA, 10.8) + [
            listed(10.8, CA), listed(16.5, CA, "RUNNING", "2026-11-20 11:15:25+01:00"), listed(22.3, CA, "GONE"),
            (24.0, CA, "system", "    GET /v1/models -> 502 Bad Gateway  (duration: 45.8 ms)"),   # after its GONE
            listed(28.1, CB)] + starting(CB, 26.7) + answering(CB, 281.8) + [
            proxy(300.0, CB, "mdf_request", kind="forwarded", method="POST", path="/v1/chat/completions", status=200),
            (300.0, CB, "container", access), (300.0, CB, "system", "    POST /v1/chat/completions -> 200 OK"),
            listed(400.0, CB, "GONE"), listed(405.0, CB, "RUNNING")]).flush()
        lv = wl.Lives().feed(path)
        self.assertEqual((lv.t_arm, lv.order, lv.overlaps(), lv.error), (t_arm, [CA, CB], [], None))
        a, b = lv.brief(CA), lv.brief(CB)
        self.assertEqual((a["id"], a["seen_s"], a["seen_by"], a["running_s"], a["gone_s"], a["last_s"]),
                         ("ta-01M7W785K…", 10.8, "line", 16.5, 22.3, 22.3))
        self.assertEqual((a["gpu"], a["vllm_s"], a["inference_lines"], a["start"]),
                         (["NVIDIA A10G"], None, 0, "2026-11-20 11:15:25+01:00"))
        self.assertEqual((b["seen_s"], b["seen_by"], b["listed_s"], b["vllm_s"], b["vllm_by"], b["ready_s"]),
                         (26.7, "line", 28.1, 281.8, "mdf_vllm_first_health", 281.8))
        self.assertEqual((b["inference_lines"], b["gone_s"], b["listed_again"]), (3, None, 1))   # listed again
        self.assertEqual(lv.alive(), [CB])
        self.assertEqual(wl.short("ta-1"), "ta-1")
        missing = wl.Lives().feed(tmp / "none")
        self.assertEqual((missing.error, missing.order), ("FileNotFoundError", []))


class TestParser(unittest.TestCase):
    def test_every_driver_names_the_app(self):
        base = ["--env-file", "e", "--server", "s", "--servers", "v"]
        for cmd in (["probe-negative"], ["probe-valid"], ["concurrency", "--input", "p"],
                    ["fault-test", "--phase", "A"], ["hold", "--marker", "m"]):
            with self.subTest(cmd=cmd[0]):
                a = mt.parser().parse_args(base + cmd + ["--app", "mdf-surya"])
                self.assertEqual((a.cmd, a.app, a.spend_ledger), (cmd[0], "mdf-surya", None))
                with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                        mock.patch("sys.stderr", io.StringIO()):
                    mt.parser().parse_args(base + cmd)



class TestModalSnapApp(unittest.TestCase):
    """MFL045 c2 (plan r2 §4): `modal_snap.py`, the app of the lever `snapshot`: a `Cls` `Serve` with the image,
    Volume, GPUs, resources and limits of `modal_app.py`, the memory snapshot and the GPU snapshot (alpha),
    `enter(snap=True)` that calls `watchdog.snap_boot`, `enter(snap=False)` that calls `watchdog.snap_wake`, and
    an empty `web_server` method on port 8000 behind the proxy token."""

    def load(self, environ, script="modal_snap.py"):
        seen = {"registry": [], "enter": []}

        class App:
            def __init__(self, name):
                seen["app"] = name

            def cls(self, **kw):
                seen["cls"] = kw
                return lambda c: c

            def function(self, **kw):
                seen["function"] = kw
                return lambda f: f

        image = mock.Mock()
        image.entrypoint.return_value = image
        image.add_local_file.return_value = image
        fake = types.ModuleType("modal")
        fake.Image = types.SimpleNamespace(
            from_registry=lambda ref, **kw: seen["registry"].append((ref, kw)) or image)
        fake.Volume = types.SimpleNamespace(from_name=lambda name, **kw: seen.setdefault("volume", (name, kw)))
        fake.App = App
        fake.concurrent = lambda **kw: (lambda c: seen.__setitem__("concurrent", kw) or c)
        fake.enter = lambda **kw: (lambda f: seen["enter"].append((f.__name__, kw)) or f)
        fake.web_server = lambda **kw: (lambda f: seen.__setitem__("web", (f.__name__, kw)) or f)
        with mock.patch.dict(sys.modules, {"modal": fake}), mock.patch.dict("os.environ", environ, clear=False):
            for k in ("MDF_FAULT", "MDF_MODAL_APP", "MDF_LEVERS"):
                if k not in environ:
                    __import__("os").environ.pop(k, None)
            spec = importlib.util.spec_from_file_location("mdf_modal_snap_t", ROOT / "scripts" / "servers" / script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        seen["chain"] = [c for c in image.mock_calls if c[0] in ("entrypoint", "add_local_file")]
        return mod, seen

    def test_the_class_and_its_options(self):
        mod, seen = self.load({"MDF_LEVERS": "snapshot,compile-cache:" + "a" * 64})
        kw = seen["cls"]
        self.assertEqual(seen["app"], "mdf-surya-snap")
        self.assertEqual((kw["enable_memory_snapshot"], kw["experimental_options"]), (True, {"enable_gpu_snapshot": True}))
        self.assertIs(type(kw["gpu"]), list)
        self.assertEqual(kw["gpu"], list(pv.MODAL_GPUS))
        self.assertEqual((kw["cpu"], kw["memory"], kw["timeout"], kw["startup_timeout"]),
                         ((2.0, 2.0), (16384, 16384), 600, 900))
        self.assertEqual((kw["scaledown_window"], kw["min_containers"], kw["max_containers"]), (60, 0, 1))
        self.assertNotIn("retries", kw)
        self.assertEqual(seen["volume"], ("mdf-surya-weights", {"create_if_missing": False}))
        self.assertEqual(kw["env"]["MDF_LEVERS"], "snapshot,compile-cache:" + "a" * 64)
        self.assertNotIn("MDF_FAULT", kw["env"])
        self.assertEqual(seen["enter"], [("boot", {"snap": True}), ("wake", {"snap": False})])
        self.assertEqual(seen["web"], ("web", {"port": 8000, "startup_timeout": 900, "requires_proxy_auth": True}))
        self.assertEqual(seen["concurrent"], {"max_inputs": 36})
        _, app = TestModalApp.load(self, {})
        self.assertEqual(seen["registry"], app["registry"])               # the image of modal_app.py
        self.assertEqual(seen["chain"][:2], app["chain"])                 # and snapshot.py, only here (F0 of MFL045)
        self.assertEqual(seen["chain"][2], mock.call.add_local_file(str(pv.HERE / "snapshot.py"), "/mdf/snapshot.py"))
        mod, seen = self.load({"MDF_MODAL_APP": "mdf-surya-snap-fault", "MDF_FAULT": "wake-fail"})
        self.assertEqual((seen["app"], seen["cls"]["env"]["MDF_FAULT"]), ("mdf-surya-snap-fault", "wake-fail"))

    def test_the_enter_methods_call_the_watchdog(self):
        mod, _ = self.load({})
        calls = []
        fake_sn = types.ModuleType("snapshot")
        fake_sn.snap_boot = lambda: calls.append("boot") or {"gen": "g"}
        fake_sn.snap_wake = lambda snap: calls.append(("wake", snap)) or "the proxy"
        with mock.patch.dict(sys.modules, {"snapshot": fake_sn}):
            s = mod.Serve()
            s.boot()
            s.wake()
            self.assertIsNone(s.web())
        self.assertEqual(calls, ["boot", ("wake", {"gen": "g"})])
        self.assertEqual(s.wd, "the proxy")


class TestSnapshotArm(unittest.TestCase):
    """MFL045 c2: the lever `snapshot` deploys `modal_snap.py` (only on Modal); its inputs are those of
    `Serve.*`, the function of the class (`modal function calls APP/Serve.*`)."""

    # The output of `modal deploy modal_snap.py` of modal 1.6.3 for the class, as on 23/11 (MFL045 F1, S0): no URL,
    # because the response of the creation does not set `method_definitions` (`modal/_output/status.py:78-93`).
    CLASS_OUT = ("✓ Created objects.\n├── ✓ Uploaded %s\n├── ✓ Uploaded %s\n├── ✓ Uploaded %s\n"
                 "└── ✓ Created Function Serve.*.\n✓ App deployed in 1.728s! 🎉\n\n"
                 "View Deployment: https://modal.com/apps/ws/main/deployed/mdf-surya-snap\n"
                 % (pv.HERE / "modal_snap.py", pv.HERE / "snapshot.py", pv.HERE / "watchdog.py"))
    SNAP_URL = "https://ws--mdf-surya-snap-serve-web.us-east.modal.run"

    @classmethod
    def class_info(cls, details):
        """`modal function info APP/Serve.* --json` of 1.6.3 (`cli/function.py` info): the methods and their web URL."""
        return json.dumps({"name": "Serve.*", "object_id": "fu-1", "app_id": "ap-1", "web_info": None,
                           "method_names": sorted(details), "method_details": details})

    def answers(self, info):
        return {"volume list": (0, json.dumps([{"name": pv.VOLUME, "created_at": "x", "created_by": "y"}])),
                "workspace proxy-tokens": (0, json.dumps([token_row(TOKEN_ID)])),
                "deploy " + str(pv.HERE / "modal_snap.py"): (0, self.CLASS_OUT),
                "function info": info}

    def test_the_url_of_the_class_from_function_info(self):
        """MFL045 F1 (S0, 23/11): the deploy of the class prints no URL; the arm reads the URL of `Serve.web` from
        `modal function info APP/Serve.* --json` while the app is deployed, within the term of the session."""
        calls = []
        info = (0, self.class_info({"web": {"web_url": self.SNAP_URL, "method": None, "unauthenticated": False}}))
        m = pv.Modal("mdf-surya-snap", runner=runner(self.answers(info), calls), environ={"PATH": "/usr/bin"},
                     env_file=env_file(self))
        self.assertEqual(m.arm("", False, pv.SNAPSHOT_LEVERS)["url"], self.SNAP_URL)
        self.assertEqual(m.url, self.SNAP_URL)
        self.assertEqual(calls[-2][0], ["deploy", str(pv.HERE / "modal_snap.py")])
        self.assertEqual(calls[-1][0], ["function", "info", "mdf-surya-snap/Serve.*", "--json"])

    def test_the_class_without_one_web_url_is_an_error(self):
        for case, info in (("no web method", (0, self.class_info({}))),
                           ("two web methods", (0, self.class_info({
                               "web": {"web_url": self.SNAP_URL, "method": None, "unauthenticated": False},
                               "other": {"web_url": URL, "method": None, "unauthenticated": False}}))),
                           ("not a modal.run URL", (0, self.class_info({
                               "web": {"web_url": "http://ws--x.example.com", "method": None,
                                       "unauthenticated": False}}))),
                           ("the CLI fails", (1, ""))):
            with self.subTest(case=case):
                m = pv.Modal("mdf-surya-snap", runner=runner(self.answers(info), []), environ={"PATH": "/usr/bin"},
                             env_file=env_file(self))
                with self.assertRaisesRegex(pv.ProviderError, "the URL of the web function"):
                    m.arm("", False, pv.SNAPSHOT_LEVERS)

    def test_the_term_during_the_reading_of_the_url(self):
        """The term T of the session that comes after the deploy, before `function info`: TermReached (S-B4), not a
        URL not found."""
        state = {"deployed": False, "reads": 0}

        def clock():
            if not state["deployed"]:
                return 1000.0
            state["reads"] += 1                                # the check after the deploy is before T, the next after
            return 1040.0 if state["reads"] == 1 else 1100.0
        answers = self.answers((0, self.class_info({"web": {"web_url": self.SNAP_URL, "method": None,
                                                                "unauthenticated": False}})))
        calls = []
        base = runner(answers, calls)

        def run(argv, **kw):
            if argv[len(pv.MODAL_SDK)] == "deploy":
                state["deployed"] = True
            return base(argv, **kw)
        m = pv.Modal("mdf-surya-snap", runner=run, clock=clock, environ={"PATH": "/usr/bin"}, env_file=env_file(self))
        with self.assertRaisesRegex(pv.TermReached, "modal function: the term of the session came during the arm"):
            m.arm("", False, pv.SNAPSHOT_LEVERS, deadline=1050.0)
        self.assertEqual(calls[-1][0][0], "deploy")             # no reading of the URL after T
        m2 = pv.Modal("mdf-surya-snap", runner=runner(answers, []), clock=lambda: 1040.0, environ={"PATH": "/usr/bin"},
                      env_file=env_file(self))
        self.assertEqual(m2.arm("", False, pv.SNAPSHOT_LEVERS, deadline=1050.0)["url"], self.SNAP_URL)

    def test_the_file_of_the_deploy_by_lever(self):
        calls = []
        out = "✓ Created web function serve.\n└── %s 🔑\n" % URL
        answers = dict(self.answers((0, self.class_info({"web": {"web_url": self.SNAP_URL, "method": None,
                                                                    "unauthenticated": False}}))),
                       **{"deploy " + str(pv.HERE / "modal_app.py"): (0, out)})
        m = pv.Modal("mdf-surya-snap", runner=runner(answers, calls), environ={"PATH": "/usr/bin"},
                     env_file=env_file(self))
        m.arm("", False, pv.SNAPSHOT_LEVERS)
        argv, env = [c for c in calls if c[0][0] == "deploy"][-1]
        self.assertEqual(argv, ["deploy", str(pv.HERE / "modal_snap.py")])
        self.assertEqual(env["MDF_LEVERS"], pv.levers_text(pv.SNAPSHOT_LEVERS))
        self.assertEqual(m.function, "Serve.*")
        m = pv.Modal("mdf-surya", runner=runner(answers, calls), environ={"PATH": "/usr/bin"}, env_file=env_file(self))
        m.arm()
        self.assertEqual(calls[-1][0], ["deploy", str(pv.HERE / "modal_app.py")])
        self.assertEqual(m.function, "serve")

    def test_the_inputs_of_the_class(self):
        calls = []
        m = pv.Modal("mdf-surya-snap", runner=runner({"function calls": (0, "[]")}, calls),
                     environ={"PATH": "/usr/bin"}, snapshot=True)
        m.inputs(1795162037.9)
        self.assertEqual(calls[0][0][:3], ["function", "calls", "mdf-surya-snap/Serve.*"])

    def test_the_levers_by_provider(self):
        self.assertEqual(pv.levers_text(pv.SNAPSHOT_LEVERS, "modal"), pv.levers_text(pv.SNAPSHOT_LEVERS))
        self.assertTrue(pv.levers_text(pv.SNAPSHOT_LEVERS).endswith(",snapshot"))
        with self.assertRaisesRegex(pv.ProviderError, "snapshot.*modal"):
            pv.levers_text("snapshot", "runpod")


class TestAfterRestore(TestRuleA):
    """MFL045 c2 (plan r2 §7): `fault-test --after-restore` on the app of the snapshot: try 1 wakes a container
    (its kind from `mdf_restore`), the driver waits for the quiet of the app (no container), then try 2; the
    phase runs on try 2 only when it is a `restore`, else the phase is inconclusive (exit 8). The container of
    try 1 is not a replacement: a declared quiet lies between the two (`next_try`)."""

    def run_after_restore(self, kinds):
        spans = [(CA, 25.0, 150.0), (CB, 175.0, 380.0)]
        lines = [proxy(25.2, CA, "mdf_restore", kind=kinds[0], gen="0123456789abcdef", age_s=0.0, pids=[23])]
        lines += starting(CA, 25.5) + answering(CA, 100.0)
        lines += [proxy(175.2, CB, "mdf_restore", kind=kinds[1], gen="0123456789abcdef", age_s=150.0, pids=[23])]
        lines += starting(CB, 175.5) + [proxy(190.0, CB, "mdf_wake", ok=True, seconds=3.1)] + answering(CB, 200.0)
        script = LogScript(self.log, self.clock, self.t_arm, Scripted.rows(spans) + lines)
        modal = Scripted(self.clock, self.t_arm, spans, script)
        routes = self.valid_routes(held=0)

        def models():
            script.flush()
            s = self.clock.t - self.t_arm
            if 100.0 <= s < 150.0 or s >= 200.0:
                return (200, {"data": [{"id": dv.MODEL}]}, "forwarded")
            return (503, {"error": "mdf_waking"}, "error")
        routes[("GET", "/v1/models")] = models
        chats = iter([(200, {"choices": []}, "forwarded"), (503, {"error": "mdf_tripped"}, "error")])
        routes[("POST", "/v1/chat/completions")] = lambda: next(chats)
        t = self.tasks(["fault-test", "--phase", "A", "--after-restore"], [mstate()], routes)
        t.modal = self.modal = modal
        with mock.patch.object(dv.Inference, "abandon", lambda self, after_s: None):     # never the network
            return self.run_quiet(t.fault_test)

    def test_the_phase_on_a_restore_after_a_creation(self):
        code, lines = self.run_after_restore(("create", "restore"))
        res = lines[-1]
        self.assertEqual(code, 0, res.get("violation"))
        self.assertTrue(res["pass"])
        self.assertEqual([(x["try"], x["kind"]) for x in res["after_restore"]], [(1, "create"), (2, "restore")])
        self.assertEqual([x["next_try"] for x in lines if "next_try" in x], [1])
        self.assertEqual(res["ready"]["snapshot"]["kind"], "restore")
        self.assertEqual(res["replacements"], [])

    def test_no_restore_in_two_tries_is_inconclusive(self):
        code, lines = self.run_after_restore(("create", "create"))
        self.assertEqual(code, 8, lines[-1])
        self.assertIn("no restore in 2 tries", lines[-1]["inconclusive"])
        self.assertNotIn("first", lines[-1])                    # the fault was never exercised


if __name__ == "__main__":
    unittest.main()
