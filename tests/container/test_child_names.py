"""Names of the children (MFL051, plan r2.1 §4.4, J.4.4): one schema, known before the creation, and a launcher
that lists, kills and removes only inside its project."""
import json
import unittest

import container_support as support
from markdown_for_llms.engine import containers


@unittest.skipUnless(support.available(), support.REASON)
class TestNames(support.ComponentCase):
    def test_the_name_is_given_to_the_observer_before_the_container_exists(self):
        seen = []

        def before(name):
            seen.append((name, self.raw_inspect(name), self.launcher.inspect(name)["state"]))
        with self.child("light") as child:
            (child.call.scratch / containers.LIMITS_FILE).write_text(json.dumps(support.CHILD_LIMITS))
            outcome = child.run("app", containers.PROBE_PYTHON, support.VARIABLES, support.LIMITS, timeout=60, before=before)
        self.assertEqual(seen, [(self.prefix + "-light-1", None, "absent")])
        self.assertEqual((outcome.cause, outcome.exit, outcome.name), ("exited", 0, self.prefix + "-light-1"))
        self.assertEqual(outcome.step()["container"], self.prefix + "-light-1")

    def test_list_is_by_the_prefix_of_the_attempt(self):
        steps = [{"do": "sleep", "seconds": 30}]
        listed = {}

        def on_poll(child, metrics):
            if not listed and self.launcher.inspect(child.name)["state"] == "running":
                listed["mine"] = self.launcher.list(self.prefix)
                listed["other"] = self.launcher.list("mdf-eng-" + "f" * 16)
                listed["docker"] = self.raw_inspect(child.name)["Config"]["Labels"]
                return "seen"
            return None
        outcome, _ = self.probe(steps=steps, on_poll=on_poll)
        self.assertEqual(listed["mine"], [{"name": self.prefix + "-light-1", "state": "running"}])
        self.assertEqual(listed["other"], [])
        self.assertEqual({k: v for k, v in listed["docker"].items() if k in ("mdf.project", "mdf.kind", "mdf.attempt")},
                         {"mdf.project": self.project, "mdf.kind": "light", "mdf.attempt": self.prefix})
        self.assertEqual((outcome.cause, outcome.reason, outcome.exit), ("stopped", "seen", 202))
        self.assertEqual(self.launcher.list(self.prefix), [])

    def test_a_prefix_or_a_name_out_of_the_schema_is_refused(self):
        for prefix in ("mdf-eng-", "mdf-eng-" + "a" * 17, "other-1", "", self.prefix + "-light-1"):
            with self.assertRaises(containers.Refused) as refused:
                self.launcher.list(prefix)
            self.assertEqual(refused.exception.code, "invalid_setting")
        for name in ("mdf-full-check", self.prefix, self.prefix + "-shell-1", "../" + self.prefix + "-light-1"):
            for op in ("inspect", "kill", "remove"):
                with self.assertRaises(containers.Refused) as refused:
                    getattr(self.launcher, op)(name)
                self.assertEqual(refused.exception.code, "invalid_setting", (op, name))

    def test_kill_and_remove_only_inside_the_project(self):
        """A container with the name of a child and without the labels of this project is not touched: here the
        launcher itself, and a container the driver creates with a child's name and another project's label."""
        name = self.name("light")
        own = self.hello["images"]["app"]["id"]
        body = {"Image": own, "Cmd": ["/bin/true"], "Labels": {"mdf.project": self.project + "-other", "mdf.kind": "light"},
                "HostConfig": {"NetworkMode": "none"}}
        status, created = self.raw("POST", "/containers/create", query={"name": name}, body=body)
        self.assertEqual(status, 201, created)
        try:
            for op in ("inspect", "kill", "remove"):
                with self.assertRaises(containers.Refused) as refused:
                    getattr(self.launcher, op)(name)
                self.assertEqual((refused.exception.code, refused.exception.detail), ("invalid_setting", "not a child of this project"))
            self.assertEqual(self.launcher.list(self.prefix), [])
            self.assertIsNotNone(self.raw_inspect(name))
            with self.assertRaises(containers.Refused):          # a name that exists cannot be taken by a child either
                self.launcher.run("light", name, "app", containers.PROBE_PYTHON, support.VARIABLES, support.LIMITS)
        finally:
            self.raw("DELETE", "/containers/%s" % created["Id"], query={"force": "1"})

    def test_remove_answers_only_when_docker_confirms_the_absence(self):
        outcome, _ = self.probe()
        self.assertEqual(outcome.exit, 0)
        self.assertEqual(self.launcher.inspect(outcome.name), {"ok": True, "state": "absent"})
        self.assertIsNone(self.raw_inspect(outcome.name))
        self.assertEqual(self.launcher.remove(outcome.name), {"ok": True, "absent": True})


if __name__ == "__main__":
    unittest.main()
