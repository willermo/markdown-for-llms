"""The term of an attempt on the wall clock, in the relay (MFL023 B2; SOL-R2 of the plan r1, SOL2-B1 of the plan r2 and
the note of Sol on the short jump before T in the review of r3).

CLOCK_MONOTONIC does not count a suspend of the laptop on Linux: after a suspend the term of the relay on the monotonic
clock has not come, while the wall clock may be past it. The relay therefore keeps the same term on the wall clock
(`until`, at most the term T of the session) and reads the drift of the wall clock on the monotonic one: past the
term, or with a drift over `servers.DRIFT_LIMIT_S` (a suspend, a step of the clock, also before T), no request of the
child leaves (503) and `check` gives `server_deadline`. The clocks are injected as a pair; a relay given a monotonic
clock alone reads no drift (the test of MFL033 `test_the_term_closes_the_relay`). The fake server is an
`httpx.MockTransport` behind the relay; the child side reaches the relay through a socketpair."""
import unittest

import httpx

from fake_inference import Fake, MODEL, PairTransport
from markdown_for_llms.engine import relay as relay_mod, servers, sockets
from markdown_for_llms.engine.codes import Code
from markdown_for_llms.engine.envfile import Secret


def endpoint():
    return servers.endpoint("rp", {"kind": "runpod", "url": "http://127.0.0.1:9", "rate_usd_per_s": "0.0002",
                                   "ready_within": 900})


class Clocks:
    """A wall clock and a monotonic clock: `tick` moves both (time that passes), `jump` the wall clock alone (a suspend
    of the laptop, or a step of its clock)."""

    def __init__(self):
        self.t_wall, self.t_mono = 1_800_000_000.0, 5000.0

    def wall(self):
        return self.t_wall

    def mono(self):
        return self.t_mono

    def tick(self, s):
        self.t_wall += s
        self.t_mono += s

    def jump(self, s):
        self.t_wall += s


class WallTerm(unittest.TestCase):
    def setUp(self):
        self.private = sockets.Private()
        self.addCleanup(self.private.close)
        self.fake = Fake(mark="forwarded")
        self.addCleanup(self.fake.close)

    def relay(self, clocks=None, **kw):
        if clocks is not None:
            kw.update(clock=clocks.mono, wall=clocks.wall)
        r = relay_mod.Relay(endpoint(), {"RUNPOD_API_KEY": Secret("rk-b2-0123456789")}, self.private.relay_socket,
                            transport_factory=self.fake.factory(), probe_interval=3600, **kw)
        self.addCleanup(r.close)
        self.r = r.start()
        return r

    def chat(self):
        client = httpx.Client(transport=PairTransport(self.r), base_url="http://relay", trust_env=False, timeout=30)
        try:
            return client.post("/v1/chat/completions", json={"model": MODEL}).status_code
        finally:
            client.close()

    def test_a_suspend_past_the_term(self):
        """Three hours of suspend with an attempt in progress: no request leaves after it."""
        clocks = Clocks()
        r = self.relay(clocks)
        r.set_term(1000)
        r.ready()
        self.assertEqual(self.chat(), 200)
        chats = self.fake.chat
        clocks.jump(3 * 3600)
        self.assertEqual(self.chat(), 503)
        self.assertEqual(self.fake.chat, chats)                  # nothing reached the server after the jump
        self.assertEqual(r.check(), Code.SERVER_DEADLINE)
        self.assertEqual(r.state, "closed")

    def test_a_short_jump_before_the_term(self):
        """Ten seconds of suspend long before T: the epoch of the term is still future, the drift is not; the request
        that runs right after the jump, before anyone reads the clocks again, does not leave."""
        clocks = Clocks()
        r = self.relay(clocks)
        r.set_term(1000, until=clocks.wall() + 3600)
        r.ready()
        chats = self.fake.chat
        clocks.jump(10)
        self.assertLess(clocks.wall(), r.until)                  # the term on the wall clock has not come
        self.assertEqual(self.chat(), 503)
        self.assertEqual(self.fake.chat, chats)
        self.assertEqual(r.check(), Code.SERVER_DEADLINE)

    def test_a_drift_within_the_limit_and_a_step_back(self):
        clocks = Clocks()
        r = self.relay(clocks)
        r.set_term(1000)
        r.ready()
        clocks.jump(3)
        self.assertEqual(self.chat(), 200)                       # a slew of NTP, not an interval lost
        self.assertIsNone(r.check())
        clocks.jump(-9)                                          # a step back of the clock: |drift| 6 s
        self.assertEqual(self.chat(), 503)
        self.assertEqual(r.check(), Code.SERVER_DEADLINE)

    def test_the_term_of_the_session_bounds_the_wall_term(self):
        clocks = Clocks()
        r = self.relay(clocks)
        r.set_term(1000, until=clocks.wall() + 100)
        self.assertEqual(r.until, clocks.wall() + 100)
        r.ready()
        clocks.tick(99)
        self.assertEqual(self.chat(), 200)
        clocks.tick(2)                                           # T passed, the term of D (995 s) not yet
        self.assertEqual(self.chat(), 503)
        self.assertEqual(r.check(), Code.SERVER_DEADLINE)

    def test_without_the_session_the_wall_term_is_the_term_of_d(self):
        clocks = Clocks()
        r = self.relay(clocks)
        r.set_term(100)
        self.assertEqual(r.until, clocks.wall() + 100 - relay_mod.CLOSE_MARGIN)

    def test_a_monotonic_clock_alone_reads_no_drift(self):
        now = [1000.0]
        r = self.relay(clock=lambda: now[0])
        r.set_term(30)
        r.ready()
        now[0] += 20.0                                           # the wall clock (real) did not move with it
        self.assertEqual(self.chat(), 200)
        self.assertIsNone(r.check())


if __name__ == "__main__":
    unittest.main()
