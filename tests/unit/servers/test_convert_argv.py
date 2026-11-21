"""The `mdf convert` of phase E against the parser of the product (MFL033 c7.18).

On 20/11 the Modal driver passed `--env-file` after `convert`, and `mdf` stopped with exit 2,
unrecognized arguments, before it could refuse with `session_orphaned`. Here the argv that each
driver hands to `subprocess.run` goes to `markdown_for_llms.cli.main` itself, not to a copy of its
parser; only `prepare_server`, the first step after the parse, is replaced: it records the parsed
arguments and refuses as an orphan window does.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "servers"))
import modal_tasks as mt  # noqa: E402
import runpod_endpoint as rpe  # noqa: E402

from markdown_for_llms import cli  # noqa: E402

URL = "https://ws--mdf-surya-fault-serve.modal.run"
NO_PID = 2 ** 22 + 12345                     # above pid_max by default: never alive


def through_mdf(cmd, seen, done=None):
    """Run the argv `cmd` of `mdf` through `cli.main(cmd[1:])`, as the `mdf` script would."""
    def prepare(args):
        seen.append(args)
        raise cli.Refused("session_orphaned", 1, cli.SESSION_HINT)
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(cli, "prepare_server", prepare), redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main(cmd[1:])
        except SystemExit as exc:
            code = exc.code
    r = subprocess.CompletedProcess(cmd, code, out.getvalue(), err.getvalue())
    if done is not None:
        done.append(r)
    return r


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.env = self.dir / ".env"
        self.input = self.dir / "input.png"
        self.input.write_bytes(b"\x89PNG\r\n\x1a\n")
        self.ledger = self.dir / "spend.jsonl"
        self.seen, self.mdf_calls, self.done = [], [], []

    def assert_parsed(self, cmd, server, servers, server_cap):
        """The parser of `mdf` took the argv of the driver, then the parsed values."""
        self.assertEqual(len(self.done), 1)
        self.assertEqual(len(self.seen), 1, "the parser of mdf refused the argv of the driver: exit %s, %s"
                         % (self.done[0].returncode, self.done[0].stderr.strip()[-300:]))
        self.assertTrue(cmd[0].endswith("/mdf"), cmd[0])
        self.assertEqual(cmd[1:4], ["--env-file", str(self.env), "convert"])
        a = self.seen[0]
        self.assertEqual((a.command, a.env_file, a.file, a.route, a.server_mode, a.confirm_transit),
                         ("convert", str(self.env), [str(self.input)], "marker-server", "balanced", True))
        self.assertEqual((a.server, a.servers, a.server_cap_usd, a.spend_ledger, a.spend_cap_usd),
                         (server, str(servers), server_cap, str(self.ledger), "15"))
        self.assertTrue(a.out.endswith("/out"))


class FakePopen:
    """The session of `hold`: it writes the marker of a valid request at once (pid never alive)."""

    def __init__(self, cmd, **kw):
        marker = Path(cmd[cmd.index("--marker") + 1])
        marker.write_text(json.dumps({"pid": NO_PID, "ready": True, "status": 200}))
        self.stdout = io.BytesIO(b"")

    def poll(self):
        return None

    def kill(self):
        pass

    def wait(self):
        return -9


class Stopped:
    """`providers.Modal` after the stop: no container."""

    def disarm(self):
        pass

    def state(self):
        return {"armed": False, "busy": False, "apps": [], "containers": [], "workers": []}


class TestModalFaultE(Base):
    def setUp(self):
        super().setUp()
        self.env.write_text("MODAL_PROXY_TOKEN_ID=wk-id\nMODAL_PROXY_TOKEN_SECRET=ws-secret\n")
        self.env.chmod(0o600)
        self.servers = self.dir / "servers.json"
        self.servers.write_text(json.dumps({"default": "modal-l4-fault", "servers": {"modal-l4-fault": {
            "kind": "modal", "url": URL, "rate_usd_per_s": "0.00036772", "gpu": "L4"}}}))

    def tasks(self, cmd):
        a = mt.parser().parse_args(["--env-file", str(self.env), "--server", "modal-l4-fault", "--servers",
                                    str(self.servers)] + cmd)
        return mt.Tasks(a, modal=Stopped(), environ={}, clock=lambda: 0.0, sleep=lambda s: None)

    def fake_run(self, cmd, **kw):
        if Path(cmd[0]).name == "mdf":
            self.mdf_calls.append(cmd)
            return through_mdf(cmd, self.seen, self.done)
        if "run" in cmd:                                          # session.py run … -- true: the orphan
            return subprocess.CompletedProcess(cmd, 3, '{"error": "session_orphaned"}', "")
        return subprocess.CompletedProcess(cmd, 0, "{}", "")      # session.py disarm

    def quiet(self, fn):
        out = io.StringIO()
        work = tempfile.mkdtemp(dir=self.dir)                     # fault_e keeps its own: not in /tmp
        with mock.patch.object(mt.subprocess, "run", self.fake_run), mock.patch.object(mt.subprocess, "Popen", FakePopen), \
                mock.patch.object(mt.tempfile, "mkdtemp", lambda prefix: work), redirect_stdout(out):
            code = fn()
        return code, [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]

    def test_fault_e_hands_mdf_an_argv_its_parser_accepts(self):
        t = self.tasks(["fault-e", "--app", "mdf-surya-fault", "--input", str(self.input), "--spend-ledger",
                        str(self.ledger), "--spend-cap-usd", "15", "--cap-usd", "0.75"])
        code, lines = self.quiet(t.fault_e)
        self.assertEqual(len(self.mdf_calls), 1)
        self.assert_parsed(self.mdf_calls[0], "modal-l4-fault", self.servers, "0.50")
        res = [x for x in lines if "fault_e" in x][-1]
        self.assertEqual((res["convert"]["exit"], res["convert"]["refused"]), (1, True))
        self.assertTrue(res["pass"])
        self.assertEqual(code, 0)

    def test_convert_refused_alone(self):
        t = self.tasks(["convert-refused", "--app", "mdf-surya-fault", "--input", str(self.input), "--spend-ledger",
                        str(self.ledger), "--spend-cap-usd", "15"])
        code, lines = self.quiet(t.convert_refused_command)
        self.assert_parsed(self.mdf_calls[0], "modal-l4-fault", self.servers, "0.50")
        self.assertEqual((code, lines[-1]["pass"], lines[-1]["convert"]["exit"]), (0, True, 1))

    def test_not_refused_is_a_violation(self):
        """A refusal with another code, or the word in an error message, is not the refusal of E."""
        t = self.tasks(["convert-refused", "--app", "mdf-surya-fault", "--input", str(self.input), "--spend-ledger",
                        str(self.ledger), "--spend-cap-usd", "15"])
        for exit_code, stdout in ((2, ""), (1, json.dumps({"status": "refused", "error": "server_not_armed"})),
                                  (0, json.dumps({"status": "refused", "error": "session_orphaned"})),
                                  (2, "mdf: error: session_orphaned")):
            with self.subTest(exit=exit_code, stdout=stdout):
                done = subprocess.CompletedProcess([], exit_code, stdout, "")
                with mock.patch.object(mt.subprocess, "run", lambda cmd, **kw: done), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(t.convert_refused_command(), 6)


class TestRunPodFaultE(Base):
    def test_convert_refused_hands_mdf_an_argv_its_parser_accepts(self):
        self.env.write_text("RUNPOD_API_KEY=rp-secret-value\n")
        self.env.chmod(0o600)
        a = rpe.parser().parse_args(["--env-file", str(self.env), "fault-e", "--name", "mdf-surya-fault", "--input",
                                     str(self.input), "--spend-ledger", str(self.ledger), "--spend-cap-usd", "15",
                                     "--cap-usd", "0.75"])
        d = rpe.Driver(a, rp=types.SimpleNamespace(url="https://ep123.api.runpod.ai"), environ={})

        def run(cmd, **kw):
            self.mdf_calls.append(cmd)
            return through_mdf(cmd, self.seen, self.done)
        with mock.patch.object(rpe.subprocess, "run", run):
            conv = d.convert_refused(str(self.dir))
        self.assert_parsed(self.mdf_calls[0], "fault", self.dir / "servers.json", "0.30")
        self.assertEqual((conv["exit"], conv["refused"]), (1, True))


class IdleEndpoint:
    """`providers.RunPod` after the disarm: max 0, no worker (the zero of the cleaning)."""
    url = "https://ep123.api.runpod.ai"

    def __init__(self):
        self.disarms = 0

    def disarm(self):
        self.disarms += 1

    def exists(self, name):
        return True

    def state(self):
        return {"armed": False, "busy": False, "summary": {"total": 0}, "workers": [], "workers_max": 0}


class TestRunPodFaultEGuarantees(Base):
    """c7.20: phase E of RunPod fails only on G2 (the refusals of `session.py run` and `mdf convert`)
    or G1 (no zero after the disarm); a valid request that failed, or the wrapped command alive after
    the SIGKILL, are facts; a session that ended before its request is inconclusive (exit 8)."""

    def setUp(self):
        super().setUp()
        self.env.write_text("RUNPOD_API_KEY=rp-secret-value\n")
        self.env.chmod(0o600)
        self.held = {"pid": NO_PID, "ready": True, "status": 200}
        self.session_again = (3, '{"error": "session_orphaned", "status": "refused"}')
        self.alive, self.killed = set(), []

    def popen(self, cmd, **kw):
        test = self

        class P(FakePopen):
            def __init__(self, cmd, **kw):
                Path(cmd[cmd.index("--marker") + 1]).write_text(json.dumps(test.held))
                self.stdout = io.BytesIO(b"")
        return P(cmd, **kw)

    def fake_run(self, cmd, **kw):
        if Path(cmd[0]).name == "mdf":
            self.mdf_calls.append(cmd)
            return through_mdf(cmd, self.seen, self.done)
        if "run" in cmd:
            return subprocess.CompletedProcess(cmd, self.session_again[0], self.session_again[1], "")
        return subprocess.CompletedProcess(cmd, 0, "{}", "")      # session.py status and disarm

    def fault_e(self):
        a = rpe.parser().parse_args(["--env-file", str(self.env), "fault-e", "--name", "mdf-surya-fault", "--input",
                                     str(self.input), "--spend-ledger", str(self.ledger), "--spend-cap-usd", "4",
                                     "--cap-usd", "0.30"])
        clock = [0.0]

        def sleep(s):
            clock[0] += s
        self.rp = IdleEndpoint()
        d = rpe.Driver(a, rp=self.rp, environ={}, clock=lambda: clock[0], sleep=sleep)
        work = tempfile.mkdtemp(dir=self.dir)
        out = io.StringIO()

        def kill(pid, sig):
            self.killed.append((pid, sig))
            self.alive.discard(pid)
        with mock.patch.object(rpe.subprocess, "run", self.fake_run), mock.patch.object(rpe.subprocess, "Popen", self.popen), \
                mock.patch.object(rpe.tempfile, "mkdtemp", lambda prefix: work), \
                mock.patch.object(rpe, "_alive", lambda pid: pid in self.alive), \
                mock.patch.object(rpe.os, "kill", kill), redirect_stdout(out):
            code = d.fault_e()
        lines = [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]
        return code, [x for x in lines if "fault_e" in x][-1], lines

    def test_passes(self):
        code, res, _ = self.fault_e()
        self.assertEqual((code, res["pass"], res["zero_s"], res["facts"]), (0, True, 0.0, []))
        self.assertEqual((res["session_again"]["exit"], res["convert"]["refused"]), (3, True))

    def test_a_failed_valid_request_is_a_fact(self):
        self.held = dict(self.held, ready=False, status=None)
        code, res, lines = self.fault_e()
        self.assertEqual((code, res["pass"]), (0, True))
        self.assertEqual([f["fact"] for f in res["facts"]], ["valid_request_failed"])
        self.assertIn("valid_request_failed", [x.get("fact") for x in lines])

    def test_the_command_alive_after_the_sigkill_is_a_fact_and_killed(self):
        self.alive.add(NO_PID)
        code, res, _ = self.fault_e()
        self.assertEqual((code, res["pass"]), (0, True))
        self.assertEqual([f["fact"] for f in res["facts"]], ["command_alive_after_sigkill"])
        self.assertEqual(self.killed[0], (NO_PID, 9))
        self.assertGreater(res["command_end_s"], 10)

    def test_no_refusal_breaks_g2(self):
        self.session_again = (0, "{}")
        code, res, _ = self.fault_e()
        self.assertEqual((code, res["guarantee"]), (6, "G2"))
        self.assertIn("session.py run did not refuse", res["violation"])
        self.assertEqual(self.rp.disarms, 1)                      # the cleaning: disarm and zero

    def test_a_session_that_ended_before_its_request_is_inconclusive(self):
        class Ended(FakePopen):
            def __init__(self, cmd, **kw):
                self.stdout = io.BytesIO(b'{"error": "provider_not_idle"}')
                self.returncode = 3

            def poll(self):
                return 3
        self.popen = lambda cmd, **kw: Ended(cmd)
        code, res, _ = self.fault_e()
        self.assertEqual(code, 8)
        self.assertIn("the session ended before the request: exit 3", res["inconclusive"])
        self.assertNotIn("violation", res)


if __name__ == "__main__":
    unittest.main()
