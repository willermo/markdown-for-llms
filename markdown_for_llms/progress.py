"""Engine progress read from the tqdm bars on the child's output (ADR 0011).

Pure module, no ML import. Marker 1.10.2 leaves the Surya 0.17.1 bars enabled and tqdm
4.70.1 writes them to stderr, which the worker already reads: `runtime.PINS` turns a
version drift into an unready worker instead of false stalls. The overall percentage is
an estimate over fixed stage bands; the phase and its count are exact.
"""
import codecs
import re
import time

# Default tqdm format up to the closing bracket of its statistics, for example
# "Recognizing Text:  25%|##5       | 12/47 [09:53<17:20, 29.7s/it]".
BAR = re.compile(r"\s*(?P<desc>[A-Za-z][A-Za-z0-9 ._-]{0,63}):\s+\d{1,3}%\|[^|\r\n]{0,80}\|\s*"
                 r"(?P<n>\d{1,8})/(?P<total>\d{1,8})\s+\[[^\[\]\r\n]{0,120}\]\s*")
SEPARATOR = re.compile(r"[\r\n]")
MAX_SEGMENT = 4096
MAX_TOTAL = 10_000_000
# Stage bands of the estimate, from the single observed log (an 11-page article on CPU).
STAGES = {"starting": (0, 5), "layout": (5, 35), "ocr-error": (35, 37), "detection": (37, 40),
          "text": (40, 85), "processors": (85, 95), "finishing": (95, 99)}
ORDER = list(STAGES)
PHASES = {"Recognizing Layout": "layout", "Running OCR Error Detection": "ocr-error",
          "Detecting bboxes": "detection", "Recognizing Text": "text", "Recognizing tables": "processors"}


class Progress:
    """Stream parser and stall clock. The clock is monotonic and never persisted;
    snapshots carry wall-clock times only."""

    def __init__(self, clock=time.monotonic, wall=time.time):
        self.clock, self.wall = clock, wall
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending, self.skipping = "", False
        self.stage, self.estimate = "starting", 0.0
        self.phase = self.current = self.total = None
        self.finished = set()
        self.events = self.unparsed = 0
        self.last = clock()
        self.max_silence = 0.0
        self.updated = self.last_advance = wall()

    def feed(self, data):
        parts = SEPARATOR.split(self.pending + self.decoder.decode(data))
        tail = parts.pop()
        if self.skipping:
            if not parts:
                return
            parts.pop(0)            # rest of an oversized segment already discarded
            self.skipping = False
        for segment in parts:
            self.segment(segment)
        # tqdm writes "\r" before a bar and nothing after it: a bar that already ends
        # with its closing bracket is complete; only a broken fragment waits.
        if BAR.fullmatch(tail):
            self.segment(tail)
            tail = ""
        elif len(tail) > MAX_SEGMENT:
            self.unparsed += 1
            self.skipping, tail = True, ""
        self.pending = tail

    def segment(self, text):
        if not text.strip():
            return
        match = BAR.fullmatch(text)
        if match is None or not int(match["n"]) <= int(match["total"]) <= MAX_TOTAL:
            self.unparsed += 1
            return
        self.observe(match["desc"], int(match["n"]), int(match["total"]))

    def observe(self, desc, n, total):
        """An advance is a new bar (another phase or total, or a restarted count) or a
        higher count; redrawing the same count is not."""
        new_bar = self.current is None or (desc, total) != (self.phase, self.total) or n < self.current
        if not new_bar and n == self.current:
            return False
        if new_bar:
            self.stage = self.next_stage(desc)
        self.phase, self.current, self.total = desc, n, total
        if total and n >= total:
            self.finished.add(self.stage)
        low, high = STAGES[self.stage]
        self.estimate = max(self.estimate, low + (high - low) * (n / total if total else 0))
        now = self.clock()
        self.max_silence = max(self.max_silence, now - self.last)
        self.last = now
        self.events += 1
        self.updated = self.last_advance = self.wall()
        return True

    def next_stage(self, desc):
        """Stages change only when a bar starts and never go back; an unknown bar
        stays in the current stage."""
        target = PHASES.get(desc)
        if target is None:
            return self.stage
        position = ORDER.index(self.stage)
        if target in ("detection", "text") and position > ORDER.index(target):
            target = "processors"
        if target == self.stage and target in self.finished and target != "processors":
            target = "processors"   # a second bar of a finished phase comes from the processors
        return target if ORDER.index(target) >= position else self.stage

    def idle(self, now=None):
        return (self.clock() if now is None else now) - self.last

    def snapshot(self):
        return dict(stage=self.stage, estimate_percent=round(self.estimate, 1), phase=self.phase,
                    current=self.current, total=self.total, updated=self.updated,
                    last_advance=self.last_advance)

    def receipt(self):
        return dict(progress=self.snapshot(), progress_events=self.events,
                    unparsed_segments=self.unparsed,
                    max_silence_seconds=max(self.max_silence, self.idle()))


def finishing(snapshot, wall=time.time):
    """The child has exited: raster validation and publication remain."""
    value = dict(stage="starting", estimate_percent=0.0, phase=None, current=None, total=None,
                 updated=None, last_advance=None)
    value.update(snapshot or {})
    value.update(stage="finishing", estimate_percent=max(float(value["estimate_percent"]), 95.0),
                 updated=wall())
    return value
