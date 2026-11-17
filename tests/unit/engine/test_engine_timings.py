"""`timings.stages` and `timings.pages` of the Marker route (M5 of MFL026, §8 of the plan), from
fake events of the Marker child: the stages between the events that open and close them, the
seconds of each page between consecutive `page` events (the first from `provider`), at most
`max_pages` values, and the monotonic clock written by the child with each event."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from markdown_for_llms.engine import core, marker_child
from markdown_for_llms.engine.pages import PAGE_TIMES, marker_timings


def events(*marks):
    return [{"event": name, "time": 1.0, "mono": at} for name, at in marks]


CHILD = events(("models", 10.0), ("provider", 14.5), ("page", 16.0), ("page", 16.5), ("page", 19.0),
               ("structure", 19.2), ("processor", 19.3), ("processor", 20.0), ("render", 21.0), ("write", 21.4))


class TimingsTests(unittest.TestCase):
    def test_stages_and_pages(self):
        timings = marker_timings(CHILD, 50)
        self.assertEqual(timings["stages"], {"models": 4.5, "pages": 4.7, "structure": 0.1, "processors": 1.7,
                                             "render": 0.4})
        pages = timings["pages"]
        self.assertEqual({k: pages[k] for k in PAGE_TIMES}, PAGE_TIMES)
        self.assertEqual((pages["values"], pages["count"], pages["median"], pages["max"]), ([1.5, 0.5, 2.5], 3, 1.5, 2.5))

    def test_at_most_max_pages_values(self):
        many = events(("provider", 0.0), *[("page", float(i)) for i in range(1, 11)])
        pages = marker_timings(many, 4)["pages"]
        self.assertEqual((pages["values"], pages["count"]), ([1.0, 1.0, 1.0, 1.0], 4))

    def test_missing_events_leave_out_what_they_would_measure(self):
        timings = marker_timings(events(("models", 1.0), ("page", 2.0), ("render", 3.0)), 50)
        self.assertEqual(timings, {"stages": {}, "pages": None})           # no provider, no write
        no_processor = marker_timings(events(("structure", 1.0), ("render", 1.5)), 50)
        self.assertEqual(no_processor["stages"], {"structure": 0.5})
        without_clock = [{"event": "provider"}, {"event": "page", "mono": "x"}, {"event": "page", "mono": True}]
        self.assertEqual(marker_timings(without_clock, 50), {"stages": {}, "pages": None})

    def test_the_child_writes_a_monotonic_clock(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(marker_child, "SCRATCH", Path(tmp)):
            marker_child.emit("page", page=1, pages=2)
            item = json.loads((Path(tmp) / "events.jsonl").read_text())
        self.assertEqual((item["event"], item["page"]), ("page", 1))
        self.assertIsInstance(item["mono"], float)

    def test_the_context_keeps_a_bounded_list_of_events(self):
        ctx = core.Context("a", Path("/nonexistent"), Path("/nonexistent"), core.DEFAULT_LIMITS, None)
        with mock.patch.object(core, "MAX_CHILD_EVENTS", 5):
            ctx.feed_events(events(*[("page", float(i)) for i in range(4)]))
            ctx.feed_events(events(*[("page", float(i)) for i in range(4)]))
        self.assertEqual(len(ctx.child_events), 5)


if __name__ == "__main__":
    unittest.main()
