"""The launcher's own watch of the children (MFL051, plan r2.1 App. H.6; SOL-R2-N6), with an injected clock.

Lease and term are measured with the monotonic clock, which does not count the time a machine is suspended, and
never with the civil time, the mtime of a file or `State.StartedAt`:
- a jump of the civil time without an advance of the monotonic clock kills nobody;
- after a restart of the launcher the lease of a child starts at its first observation, and its term is read from
  the label `mdf.t0`;
- a label in the future (the clock restarted with the host) is an expired term.

The launcher is the outer limit of the term (SOL-A1-B2): the time it leaves to the supervisor of a child has an
absolute cap, and a child it kills is reported with the launcher's reason, never as a memory limit. Its stop and
the creations are coordinated (SOL-A1-B1), and the stop ends within its budget with a slow daemon too (SOL-A1-N3):
those races are driven here with events, never with a pause.
"""
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest import mock

import launcher_support as support
from markdown_for_llms import launcher
from markdown_for_llms.engine import containers, toolchain
from markdown_for_llms.engine.codes import Code

NAME = support.ATTEMPT + "-light-1"
G = launcher.GRACE_SECONDS              # the watch acts this long after the supervisor of the child should have
CAP = 30.0                              # and never longer than this, whatever the term (`launcher.TERM_GRACE_MAX`)
CALL_SECONDS = launcher.Docker("/nowhere").timeout     # what a call to Docker waits when nobody gives it a timeout


class WatchCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        self.sut, self.fake, self.clock = support.make(roots={"work": str(self.work)})
        self.watch = launcher.Watch(self.sut, clock=self.clock)

    def child(self, name=NAME, t0=None, term=120, lease=30, state="running", project=support.PROJECT, labels=None):
        t0 = self.clock.now if t0 is None else t0
        wanted = {"mdf.project": project, "mdf.kind": "light", "mdf.attempt": support.ATTEMPT, "mdf.max": str(term),
                  "mdf.lease": str(lease), "mdf.t0": repr(float(t0))}
        self.fake.add(name, wanted if labels is None else labels, state=state)
        (self.work / "call" / name / "ctl").mkdir(parents=True, exist_ok=True)
        self.renew(name, 0)
        return name

    def renew(self, name, counter):
        (self.work / "call" / name / "ctl" / "lease").write_text("%d\n" % counter)

    def tick(self, seconds=0.0):
        self.clock.now += seconds
        return self.watch.tick()

    def gone(self, name=NAME):
        return self.fake.state(name) is None


class TestLease(WatchCase):
    def test_a_child_that_renews_lives_and_one_that_does_not_is_killed_and_removed(self):
        self.child(term=1000)
        for counter in range(1, 100):
            self.renew(NAME, counter)
            self.assertEqual(self.tick(1.0), {})
        self.assertEqual(self.fake.state(NAME), "running")
        self.assertEqual(self.tick(30.0), {})                   # the lease: the supervisor of the child ends it now
        self.assertEqual(self.tick(G), {})                      # and the watch gives it the time to
        self.assertEqual(self.tick(1.0), {NAME: "lease"})
        self.assertTrue(self.gone())
        self.assertEqual(len(self.fake.did("POST", "/kill")), 1)
        self.assertEqual(len(self.fake.did("DELETE", "/containers/")), 1)

    def test_a_jump_of_the_civil_time_kills_nobody(self):
        self.child()
        self.assertEqual(self.tick(), {})
        for jump in (10 * 365 * 86400, -10 * 365 * 86400):
            with mock.patch.object(time, "time", return_value=time.time() + jump):
                self.assertEqual(self.tick(1.0), {})
        self.assertEqual(self.fake.state(NAME), "running")

    def test_the_watch_never_reads_the_civil_time_or_the_time_of_a_file(self):
        self.child()
        with mock.patch.object(time, "time", side_effect=AssertionError("civil time read")):
            self.assertEqual(self.tick(1.0), {})
            os.utime(self.work / "call" / NAME / "ctl" / "lease", (0, 0))          # an old mtime is not an old lease
            self.assertEqual(self.tick(1.0), {})
            os.utime(self.work / "call" / NAME / "ctl" / "lease", (2 ** 33, 2 ** 33))
            self.assertEqual(self.tick(40.0), {NAME: "lease"})                     # and a new mtime is not a renewal
        source = Path(launcher.__file__).read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", source, flags=re.S)
        self.assertNotRegex(code, r"time\.time\(|st_mtime|datetime|getmtime")
        self.assertEqual(len(re.findall(r"StartedAt|FinishedAt", code)), 2)        # passed on by `inspect`, never compared

    def test_after_a_restart_the_lease_starts_at_the_first_observation(self):
        """The memory of the leases is empty in a new launcher: a child found with a counter that stopped long ago
        gets a whole lease from the instant it is first seen (inside it, its supervisor lost nothing)."""
        self.child(t0=self.clock.now - 100, term=1000)
        self.renew(NAME, 41)
        restarted = launcher.Watch(self.sut, clock=self.clock)
        self.assertEqual(restarted.tick(), {})
        self.clock.now += 30.0 + G
        self.assertEqual(restarted.tick(), {})
        self.clock.now += 0.5
        self.assertEqual(restarted.tick(), {NAME: "lease"})

    def test_a_renewal_after_the_restart_counts_from_there(self):
        self.child(t0=self.clock.now - 100, term=1000)
        restarted = launcher.Watch(self.sut, clock=self.clock)
        self.assertEqual(restarted.tick(), {})
        self.clock.now += 29.0
        self.renew(NAME, 7)
        self.assertEqual(restarted.tick(), {})
        self.clock.now += 29.0 + G
        self.assertEqual(restarted.tick(), {})
        self.clock.now += 2.0
        self.assertEqual(restarted.tick(), {NAME: "lease"})

    def test_a_lease_that_cannot_be_read_is_a_lease_that_does_not_change(self):
        """A FIFO, a link or nothing in place of `ctl/lease`: read as absent, without blocking, and never a renewal."""
        for n, make in enumerate(("fifo", "link", "absent", "directory")):
            name = self.child(name="%s-light-%d" % (support.ATTEMPT, 10 + n), term=100000)
            lease = self.work / "call" / name / "ctl" / "lease"
            lease.unlink()
            if make == "fifo":
                os.mkfifo(lease)
            elif make == "link":
                (self.work / "elsewhere").write_text("1\n")
                os.symlink(self.work / "elsewhere", lease)
            elif make == "directory":
                lease.mkdir()
            started = time.monotonic()
            self.assertEqual(self.tick(), {})
            (self.work / "elsewhere").write_text("2\n") if make == "link" else None
            self.assertEqual(self.tick(31.0 + G), {name: "lease"}, make)
            self.assertLess(time.monotonic() - started, 2)

    def test_a_child_whose_folder_is_not_seen_has_a_lease_that_does_not_change(self):
        sut, fake, clock = support.make()                       # a launcher without the work volume: no lease to read
        fake.add(NAME, {"mdf.project": support.PROJECT, "mdf.kind": "manifest", "mdf.attempt": "-", "mdf.max": "60",
                        "mdf.lease": "60", "mdf.t0": repr(clock.now)})
        watch = launcher.Watch(sut, clock=clock)
        self.assertEqual(watch.tick(), {})
        clock.now += 59 + G
        self.assertEqual(watch.tick(), {})
        clock.now += 2
        self.assertEqual(watch.tick(), {NAME: "term"})


class TestTerm(WatchCase):
    def test_the_term_is_counted_from_the_label_even_when_the_lease_is_renewed(self):
        self.child(term=120)
        for counter in range(1, 121):
            self.renew(NAME, counter)
            self.assertEqual(self.tick(1.0), {})
        self.renew(NAME, 999)
        self.assertEqual(self.tick(G - 1.0), {})                # the supervisor ends the child at its term; the watch waits
        self.renew(NAME, 1000)
        self.assertEqual(self.tick(2.0), {NAME: "term"})
        self.assertTrue(self.gone())

    def test_after_a_restart_the_term_is_read_from_the_label(self):
        self.child(t0=self.clock.now - 119, term=120)
        restarted = launcher.Watch(self.sut, clock=self.clock)
        self.assertEqual(restarted.tick(), {})
        self.clock.now += 2 + G
        self.assertEqual(restarted.tick(), {NAME: "term"})

    def test_a_label_in_the_future_is_an_expired_term(self):
        """After a reboot of the host the monotonic clock restarts from zero: the instant of a child of before is
        in the future. Such a container is not running any more; if one is, it is past its term."""
        self.child(t0=self.clock.now + 5000, term=86400)
        self.assertEqual(self.tick(), {NAME: "term"})
        self.assertTrue(self.gone())

    def test_a_child_without_its_labels_is_past_its_term(self):
        for n, broken in enumerate(({"mdf.max": "x"}, {"mdf.t0": "soon"}, {"mdf.lease": ""}, {"mdf.t0": None})):
            name = "%s-light-%d" % (support.ATTEMPT, 20 + n)
            labels = {"mdf.project": support.PROJECT, "mdf.kind": "light", "mdf.attempt": support.ATTEMPT, "mdf.max": "120",
                      "mdf.lease": "30", "mdf.t0": repr(self.clock.now)}
            labels.update(broken)
            self.child(name=name, labels={k: v for k, v in labels.items() if v is not None})
            self.assertEqual(self.tick(), {name: "term"}, broken)


class TestEndedAndOthers(WatchCase):
    def test_a_child_that_exited_is_removed_after_ten_minutes(self):
        self.child(state="exited", term=3600)
        self.assertEqual(self.tick(), {})
        self.assertEqual(self.tick(600.0), {})
        self.assertEqual(self.tick(1.0), {NAME: "ended"})
        self.assertTrue(self.gone())
        self.assertEqual(self.fake.did("POST", "/kill")[-1][0], "POST")

    def test_an_exited_child_of_before_a_reboot_or_long_past_its_term_is_removed_at_once(self):
        self.child(name=NAME, state="exited", t0=self.clock.now + 9000)
        other = self.child(name=support.ATTEMPT + "-light-2", state="exited", t0=self.clock.now - 800, term=120)
        fresh = self.child(name=support.ATTEMPT + "-light-3", state="exited", t0=self.clock.now - 130, term=120)
        self.assertEqual(self.tick(), {NAME: "ended", other: "ended"})
        self.assertEqual(self.fake.state(fresh), "exited")

    def test_a_container_just_created_is_not_a_running_child(self):
        self.child(state="created")
        self.assertEqual(self.tick(40.0), {})
        self.assertEqual(self.fake.state(NAME), "created")

    def test_docker_unreachable_changes_nothing(self):
        self.child()
        self.assertEqual(self.tick(), {})
        self.fake.fail = "ECONNREFUSED"
        self.assertEqual(self.tick(100.0), {})
        self.fake.fail = None
        self.assertEqual(self.tick(), {NAME: "lease"})          # the memory of the lease was kept

    def test_the_containers_of_other_projects_are_never_touched(self):
        self.child(name=support.ATTEMPT + "-light-7", project="another-project", t0=self.clock.now - 99999)
        self.fake.add("somebody-else", {"com.docker.compose.project": support.PROJECT})
        self.fake.add("no-labels", {})
        self.assertEqual(self.tick(99999.0), {})
        self.assertEqual(len(self.fake.containers), 3)
        self.assertEqual(self.fake.did("POST", "/kill") + self.fake.did("DELETE", "/containers/"), [])

    def test_the_memory_forgets_the_children_that_are_gone(self):
        self.child()
        self.tick()
        self.assertIn(NAME, self.watch.leases)
        del self.fake.containers[self.fake._find(NAME)]
        self.tick()
        self.assertEqual((self.watch.leases, self.watch.ended), ({}, {}))


class TestStop(WatchCase):
    def test_sigterm_refuses_run_and_removes_the_children(self):
        self.child()
        self.child(name=support.ATTEMPT + "-light-2", state="exited")
        self.fake.add("of-another-project", {"mdf.project": "other", "mdf.kind": "light"})
        self.assertTrue(self.sut.stop())
        self.assertEqual(list(c["name"] for c in self.fake.containers.values()), ["of-another-project"])
        answer, _ = self.sut.handle(support.request())
        self.assertEqual((answer["code"], answer["detail"]), ("isolation_unavailable", "the launcher is stopping"))

    def test_a_stop_without_docker_gives_up_and_says_so(self):
        self.child()
        self.fake.fail = "ENOENT"
        self.assertFalse(self.sut.stop())


class TestGrace(WatchCase):
    def test_the_grace_of_a_term_is_a_share_of_it(self):
        """Lease and term of the supervisor are counted in ticks, and a tick may last a little more than its tenth
        of a second: for a term of minutes the watch waits two per cent more, so that the supervisor is still first."""
        self.child(term=1000, lease=100000)
        self.assertEqual(self.tick(1000 + 19), {})
        self.assertEqual(self.tick(2), {NAME: "term"})

    def test_the_grace_has_an_absolute_cap(self):
        """SOL-A1-B2. Two per cent of a day would be 28 minutes: the launcher is the outer limit of the term, so
        the time it leaves to the supervisor is never more than `TERM_GRACE_MAX`, whatever the term."""
        for n, term in enumerate((1500, 3600, 86400)):
            name = "%s-light-%d" % (support.ATTEMPT, 30 + n)
            self.child(name=name, term=term, lease=300)
            start = self.clock.now
            while self.clock.now - start < term + CAP - 100:        # a caller that renews, to the end
                self.clock.now += 100
                self.renew(name, int(self.clock.now))
                self.assertEqual(self.watch.tick(), {}, term)
            self.clock.now = start + term + CAP - 1
            self.renew(name, int(self.clock.now))
            self.assertEqual(self.watch.tick(), {}, term)
            self.assertEqual(self.fake.state(name), "running")
            self.renew(name, int(self.clock.now) + 1)
            self.assertEqual(self.tick(2), {name: "term"}, term)
            self.assertTrue(self.gone(name))
        self.assertEqual(launcher.TERM_GRACE_MAX, CAP)

    def test_no_term_a_request_admits_is_left_more_than_the_cap(self):
        for term in (1, 60, 250, 1499, 1500, 1501, 7200, launcher.MAXIMA["max_seconds"]):
            name = "%s-light-%d" % (support.ATTEMPT, term)
            self.child(name=name, term=term, lease=100000)
            self.assertEqual(self.tick(term + CAP + 1), {name: "term"}, term)


class TestReason(WatchCase):
    """SOL-A1-B2. A child the launcher kills reads as 137, like a child killed for memory, or as nothing at all once
    it is removed. The launcher says why in `inspect`, and the client reports that reason."""

    def inspect(self, name=NAME):
        answer, _ = self.sut.handle({"op": "inspect", "name": name})
        return answer

    def keep_the_killed(self):
        """Docker kills and does not remove: what a caller sees between the two requests of the launcher."""
        route = self.fake._route

        def keeps(method, path, query, body):
            return (500, {"message": "busy"}) if method == "DELETE" else route(method, path, query, body)
        self.fake._route = keeps

    def test_a_child_killed_at_its_term_says_so_once_removed(self):
        self.child(term=120, lease=100000)
        self.assertEqual(self.inspect()["state"], "running")
        self.assertNotIn("reaped", self.inspect())
        self.assertEqual(self.tick(120 + G + 1), {NAME: "term"})
        self.assertEqual(self.inspect(), {"ok": True, "state": "absent", "reaped": "term"})

    def test_a_child_killed_at_its_lease_says_so_while_its_137_is_still_there(self):
        self.keep_the_killed()
        self.child(term=100000, lease=30)
        self.assertEqual(self.tick(), {})
        self.assertEqual(self.tick(30 + G + 1), {NAME: "lease"})
        answer = self.inspect()
        self.assertEqual((answer["state"], answer["exit"], answer["oom_killed"], answer["reaped"]), ("exited", 137, False, "lease"))

    def test_the_reason_is_noted_before_the_kill(self):
        """No `inspect` can show the 137 of the launcher's own kill without its reason."""
        self.child(term=120, lease=100000)
        seen = []
        route = self.fake._route

        def at_the_kill(method, path, query, body):
            if method == "POST" and path.endswith("/kill"):
                seen.append(dict(self.sut.reaped))
            return route(method, path, query, body)
        self.fake._route = at_the_kill
        self.tick(120 + G + 1)
        self.assertEqual([{name: why for name, (why, _) in noted.items()} for noted in seen], [{NAME: "term"}])

    def test_an_exited_child_removed_later_and_the_stop_of_the_launcher_have_their_reasons(self):
        self.child(state="exited", term=3600)
        self.tick()
        self.assertEqual(self.tick(601.0), {NAME: "ended"})
        self.assertEqual(self.inspect()["reaped"], "ended")
        other = self.child(name=support.ATTEMPT + "-light-2")
        self.keep_the_killed()
        self.sut.stop(seconds=1.0)
        self.assertEqual((self.inspect(other)["exit"], self.inspect(other)["reaped"]), (137, "stop"))
        self.assertEqual(set(why for why, _ in self.sut.reaped.values()) - set(containers.REAPED), set())

    def test_a_new_child_of_the_same_name_has_no_reason_of_the_old_one(self):
        work = self.work / "call" / NAME
        for role in ("scratch", "in", "ctl", "stat"):
            (work / role).mkdir(parents=True)
        sut, fake, clock = support.make(roots={"work": str(self.work)}, uid=os.getuid())
        sut.note(NAME, "term")
        self.assertEqual(sut.handle({"op": "inspect", "name": NAME})[0], {"ok": True, "state": "absent", "reaped": "term"})
        self.assertTrue(sut.handle(support.request())[0]["ok"])
        self.assertNotIn("reaped", sut.handle({"op": "inspect", "name": NAME})[0])

    def test_the_reasons_are_forgotten_after_ten_minutes(self):
        self.sut.note(NAME, "term")
        self.clock.now += launcher.REAPED_SECONDS + 1
        self.sut.note(support.ATTEMPT + "-light-2", "lease")
        self.assertEqual(list(self.sut.reaped), [support.ATTEMPT + "-light-2"])

    def test_a_term_of_a_day_ends_within_the_cap_with_the_code_of_a_timeout(self):
        """The whole path, with the clock of the test: a caller that renews its lease for the longest term a
        request admits, a child whose supervisor does nothing, the launcher's watch once a second of its clock.
        The child is gone within the cap, and the caller reports a timeout, not a memory limit and not a failure."""
        term = launcher.MAXIMA["max_seconds"]
        sut, fake, clock = support.make(roots={"work": str(self.work)}, uid=os.getuid())
        watch = launcher.Watch(sut, clock=clock)

        class Direct:
            """The launcher without its socket: what `containers.Launcher` would answer."""
            def call(self, request):
                answer, _ = sut.handle(request)
                if not answer["ok"]:
                    raise (containers.Unreachable if answer["code"] == "isolation_unavailable" else containers.Refused)(answer["detail"])
                return answer

            def run(self, kind, name, image, argv, variables, limits):
                return self.call({"op": "run", "kind": kind, "name": name, "image": image, "argv": list(argv), "env": dict(variables),
                                  "limits": dict(limits)})

            def inspect(self, name):
                return self.call({"op": "inspect", "name": name})

            def remove(self, name):
                return self.call({"op": "remove", "name": name})

        def sleep(seconds):                     # the clock of the caller: while it waits, the watch of the launcher works
            clock.now += seconds
            watch.tick()
        child = containers.Child(Direct(), self.work, NAME, clock=clock, sleep=sleep)
        request = support.request()
        with child:
            outcome = child.run("app", request["argv"], request["env"], dict(support.LIMITS, max_seconds=term, lease_seconds=300),
                                poll_seconds=20.0)
        self.assertEqual((outcome.cause, outcome.exit, outcome.reason), ("term", None, None))
        self.assertEqual(containers.error_of(outcome, "what").code, Code.TIMEOUT)
        self.assertEqual(containers.error_of(outcome, "what", tool=True).code, Code.TOOL_TIMEOUT)
        self.assertGreater(outcome.seconds, term)
        self.assertLessEqual(outcome.seconds, term + CAP + 20.0)
        self.assertEqual((fake.containers, child.removed), ({}, True))


class Asked:
    """The lock of the creations, saying each time somebody asks for it (the tests wait for that, not for a time)."""

    def __init__(self, lock):
        self.lock, self.asked = lock, threading.Semaphore(0)

    def acquire(self, *args, **kw):
        self.asked.release()
        return self.lock.acquire(*args, **kw)

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class TestStopAndCreation(unittest.TestCase):
    """SOL-A1-B1. After the stop of the launcher began nothing is created, and what was being created is within its
    cleanup. Every race is held open with an event and closed by the test: no pause decides the order."""

    WAIT = 10.0                             # a bound for a thread that must not hang, never a pause that orders two events

    def setUp(self):
        self.sut, self.fake, self.clock = support.make()
        self.answers, self.stopped = [], []

    def start(self, target):
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    def run_request(self, request=None):
        return self.start(lambda: self.answers.append(self.sut.handle(request or support.request())[0]))

    def stop(self):
        return self.start(lambda: self.stopped.append(self.sut.stop()))

    def end(self, *threads):
        for thread in threads:
            thread.join(self.WAIT)
            self.assertFalse(thread.is_alive())

    def refused(self):
        self.assertEqual(len(self.answers), 1)
        answer = self.answers[0]
        self.assertEqual((answer["ok"], answer.get("code"), answer.get("detail")), (False, "isolation_unavailable", "the launcher is stopping"),
                         "a child was created or kept after the stop: %s" % sorted(c["name"] for c in self.fake.containers.values()))

    def hold(self, where):
        """Hold the request to Docker that `where(method, path)` chooses; returns (entered, release)."""
        entered, release = threading.Event(), threading.Event()
        route = self.fake._route

        def held(method, path, query, body):
            if where(method, path):
                entered.set()
                self.assertTrue(release.wait(self.WAIT), "the test did not release the request it held")
            return route(method, path, query, body)
        self.fake._route = held
        return entered, release

    def test_a_creation_suspended_while_its_image_is_resolved_creates_nothing_after_the_stop(self):
        """The case of the review: the request passed the check at its entry, the stop came and found nothing, and
        the request went on to create and start a child that nobody removed."""
        entered, release = threading.Event(), threading.Event()
        image = self.sut.image

        def waits(*args):
            entered.set()
            self.assertTrue(release.wait(self.WAIT))
            return image(*args)
        self.sut.image = waits
        request = self.run_request()
        self.assertTrue(entered.wait(self.WAIT))
        self.assertIs(self.sut.stop(), True)
        release.set()
        self.end(request)
        self.refused()
        self.assertEqual(self.fake.did("POST", "/containers/create"), [])
        self.assertEqual((self.fake.containers, self.sut.children()), ({}, []))

    def test_the_stop_waits_for_a_creation_in_flight_and_leaves_nothing(self):
        """Docker is answering `create` when the stop comes: the stop waits for the lock of the creations, and the
        creation, which sees the stop before its start, removes what it created."""
        self.sut.creating = lock = Asked(self.sut.creating)
        entered, release = self.hold(lambda method, path: (method, path) == ("POST", "/containers/create"))
        request = self.run_request()
        self.assertTrue(entered.wait(self.WAIT))
        self.assertTrue(lock.asked.acquire(timeout=self.WAIT))                  # the creation took the lock
        stop = self.stop()
        self.assertTrue(lock.asked.acquire(timeout=self.WAIT), "the stop did not wait for the creation in flight")
        self.assertEqual((self.stopped, self.answers), ([], []))                # both wait: the stop for the lock
        release.set()
        self.end(request, stop)
        self.refused()
        self.assertEqual(self.stopped, [True])
        self.assertEqual(len(self.fake.did("POST", "/containers/create")), 1)
        self.assertEqual(self.fake.did("POST", "/start"), [])                   # created, never started, removed
        self.assertEqual(self.fake.containers, {})

    def test_a_child_whose_start_was_in_flight_is_removed_by_the_stop(self):
        """The stop comes while Docker starts the child: it is listed when the stop has the lock, and removed; the
        creation sees the stop after its start and removes it too, whichever comes first."""
        self.sut.creating = lock = Asked(self.sut.creating)
        entered, release = self.hold(lambda method, path: method == "POST" and path.endswith("/start"))
        request = self.run_request()
        self.assertTrue(entered.wait(self.WAIT))
        self.assertTrue(lock.asked.acquire(timeout=self.WAIT))
        stop = self.stop()
        self.assertTrue(lock.asked.acquire(timeout=self.WAIT), "the stop did not wait for the creation in flight")
        release.set()
        self.end(request, stop)
        self.refused()
        self.assertEqual((self.stopped, self.fake.containers), ([True], {}))

    def test_the_stop_that_comes_during_the_start_is_seen_after_it(self):
        """Without any thread: the stop is set while Docker starts the child (SIGTERM sets it from the handler)."""
        self.fake.on_start = lambda container: self.sut.stopping.set()
        self.answers.append(self.sut.handle(support.request())[0])
        self.refused()
        self.assertEqual(len(self.fake.did("POST", "/start")), 1)
        self.assertEqual(self.fake.containers, {})

    def test_after_the_stop_every_creation_is_refused_at_the_serialized_point(self):
        """The check at the entry of `run` is not the one that holds: with it bypassed, as a request that had
        already passed it, the creation is refused where it is serialized, immediately before the request to Docker."""
        self.assertIs(self.sut.stop(), True)
        self.assertEqual(self.sut.handle(support.request())[0]["detail"], "the launcher is stopping")
        with self.assertRaises(launcher.Refusal) as refused:
            self.sut._create({"name": support.ATTEMPT + "-light-1"}, support.body_for(support.request()))
        self.assertEqual((refused.exception.code, refused.exception.detail), ("isolation_unavailable", "the launcher is stopping"))
        answer, _ = self.sut.handle({"op": "toolchain"})            # the launcher's own manifest child is a creation too
        self.assertEqual((answer["ok"], answer["detail"]), (False, "the launcher is stopping"))
        self.assertEqual(self.fake.did("POST", "/containers/create"), [])

    def test_the_manifest_child_of_the_launcher_is_created_under_the_same_lock(self):
        self.sut.creating = lock = Asked(self.sut.creating)
        entered, release = self.hold(lambda method, path: (method, path) == ("POST", "/containers/create"))
        support.manifest_exits(self.fake, b"f 00  /usr/bin/x\n")
        request = self.run_request({"op": "toolchain"})
        self.assertTrue(entered.wait(self.WAIT))
        self.assertTrue(lock.asked.acquire(timeout=self.WAIT), "the manifest child was created outside the lock of the creations")
        stop = self.stop()
        self.assertTrue(lock.asked.acquire(timeout=self.WAIT), "the stop did not wait for the creation in flight")
        release.set()
        self.end(request, stop)
        self.refused()
        self.assertEqual((self.stopped, self.fake.containers, self.sut.verified), ([True], {}, set()))

    def test_a_verification_in_flight_ends_at_the_stop(self):
        """The manifest child runs when the stop comes: the stop removes it, and the verification does not wait
        its ninety seconds for a child that is gone."""
        self.fake.on_start = lambda container: None                 # it starts and never exits
        entered, release = self.hold(lambda method, path: method == "GET" and re.fullmatch(r"/containers/[0-9a-f]{64}/json", path)
                                     and self.fake.did("POST", "/start"))
        request = self.run_request({"op": "toolchain"})
        self.assertTrue(entered.wait(self.WAIT))                    # the verification polls its child
        started = self.clock.now
        self.assertIs(self.sut.stop(), True)
        release.set()
        self.end(request)
        self.refused()
        self.assertEqual(self.fake.containers, {})
        self.assertLess(self.clock.now - started, 5.0)


class TestStopBudget(unittest.TestCase):
    """SOL-A1-N3. The budget of the stop is the timeout of every call it makes to Docker, never more than what is
    left: a daemon that is slow, or silent, cannot make the stop last longer than `TERM_SECONDS`."""

    def setUp(self):
        self.sut, self.fake, self.clock = support.make()
        for n in (1, 2, 3):
            self.fake.add("%s-light-%d" % (support.ATTEMPT, n), {"mdf.project": support.PROJECT, "mdf.kind": "light",
                                                                 "mdf.attempt": support.ATTEMPT})
        self.timeouts = []

    def daemon(self, seconds, answers=True):
        """A daemon that takes `seconds` of the clock for every call, and gives up at the timeout it is given, as a
        real call does (the thirty seconds of `Docker` without one)."""
        call = self.fake.call

        def slow(method, path, **kw):
            left = self.until - self.clock.now
            timeout = kw.get("timeout") or CALL_SECONDS
            self.timeouts.append((timeout, left))
            self.clock.sleep(min(seconds, timeout))
            if not answers or seconds > timeout:
                raise launcher.DockerUnreachable("no answer")
            return call(method, path, **kw)
        self.fake.call = slow

    def stop(self, seconds=launcher.TERM_SECONDS):
        started = self.clock.now
        self.until = started + seconds
        return self.sut.stop(seconds) if seconds != launcher.TERM_SECONDS else self.sut.stop(), self.clock.now - started

    def test_a_silent_daemon_does_not_make_the_stop_longer_than_its_budget(self):
        """The case of the review: one call that never answers lasted its own thirty seconds, past the twenty-five of the stop."""
        self.daemon(3600, answers=False)
        clean, elapsed = self.stop()
        self.assertEqual((clean, elapsed), (False, launcher.TERM_SECONDS))
        self.assertEqual(self.timeouts, [(launcher.TERM_SECONDS, launcher.TERM_SECONDS)])

    def test_every_call_gets_what_is_left_and_never_more(self):
        self.daemon(2.0)
        clean, elapsed = self.stop()
        self.assertIs(clean, True)                                  # list, three kills and removals, list again: 16.2 s
        self.assertLessEqual(elapsed, launcher.TERM_SECONDS)
        self.assertEqual(self.fake.containers, {})
        self.assertGreater(len(self.timeouts), 7)
        for timeout, left in self.timeouts:
            self.assertEqual(timeout, min(left, CALL_SECONDS))
            self.assertGreater(left, 0)

    def test_a_daemon_too_slow_for_all_the_children_stops_the_cleanup_at_the_end_of_the_budget(self):
        self.daemon(6.0)
        clean, elapsed = self.stop()
        self.assertIs(clean, False)
        self.assertLessEqual(elapsed, launcher.TERM_SECONDS)
        self.assertLess(self.timeouts[-1][0], 6.0)                  # the last call was given less than it needed
        self.assertEqual(len(self.fake.containers), 2)              # one child went, the others stay under their supervisors
        for timeout, left in self.timeouts:
            self.assertLessEqual(timeout, left)

    def test_a_longer_budget_is_still_cut_to_the_timeout_of_a_call(self):
        self.daemon(3600, answers=False)
        clean, elapsed = self.stop(seconds=100.0)
        self.assertEqual((clean, elapsed, self.timeouts), (False, CALL_SECONDS, [(CALL_SECONDS, 100.0)]))

    def test_no_budget_at_all(self):
        self.daemon(1.0)
        clean, elapsed = self.stop(seconds=0.0)
        self.assertEqual((clean, elapsed, self.timeouts), (False, 0.0, []))

    def test_the_watch_and_the_requests_keep_the_timeout_of_a_call(self):
        """Outside the stop nothing has a budget: the calls keep the default timeout of `Docker`."""
        seen = []
        call = self.fake.call

        def spy(method, path, **kw):
            seen.append(kw.get("timeout"))
            return call(method, path, **kw)
        self.fake.call = spy
        launcher.Watch(self.sut, clock=self.clock).tick()
        self.sut.handle({"op": "inspect", "name": support.ATTEMPT + "-light-1"})
        self.assertEqual(set(seen), {None})
        self.assertEqual((CALL_SECONDS, launcher.DOCKER_SECONDS), (30.0, 30.0))


if __name__ == "__main__":
    unittest.main()
