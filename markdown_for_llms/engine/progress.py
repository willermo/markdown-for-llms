"""Progress of a conversion from the events of the engine (MFL032; replaces the tqdm parser).

The Marker child writes one JSON event per step (`marker_child.emit`): models, provider, one
per page read by Indra, structure, one per processor, render, write. `Progress.feed(event)`
turns them into the snapshot that the worker stores and the Web UI shows, with the same keys
as before: stage, estimate_percent, phase, current, total, updated, last_advance. The overall
percentage is an estimate over fixed bands; the phase and its page count are exact. Pure
module, standard library only (the API process imports `STAGES`).
"""
import time

STAGES = {"starting": (0, 5), "preparing": (5, 10), "models": (10, 15), "pages": (15, 85),
          "processors": (85, 95), "finishing": (95, 99)}
ORDER = list(STAGES)
EVENT_STAGE = {"prepare": "preparing", "models": "models", "provider": "pages", "page": "pages",
               "structure": "processors", "processor": "processors", "render": "processors",
               "write": "finishing", "publish": "finishing"}
# the phase shown with the estimate (the label of the CLI and of the Web UI)
PHASES = {"prepare": "Preparing the PDF", "models": "Loading the models", "provider": "Opening the PDF",
          "page": "Reading pages", "structure": "Building the structure", "render": "Rendering",
          "write": "Writing", "publish": "Publishing"}


class Progress:
    def __init__(self, wall=time.time):
        self.wall = wall
        self.stage, self.estimate = "starting", 0.0
        self.phase = self.current = self.total = None
        self.events = 0
        self.updated = self.last_advance = wall()

    def feed(self, event):
        """One event of the engine; stages never go back. Returns True when it advanced."""
        name = event.get("event")
        stage = EVENT_STAGE.get(name)
        if stage is None:
            return False
        if ORDER.index(stage) < ORDER.index(self.stage):
            stage = self.stage
        self.stage = stage
        low, high = STAGES[stage]
        if name == "page" and isinstance(event.get("page"), int) and isinstance(event.get("pages"), int) \
                and 0 < event["page"] <= event["pages"]:
            self.phase, self.current, self.total = PHASES["page"], event["page"], event["pages"]
            share = event["page"] / event["pages"]
        else:
            label = event.get("name") if name == "processor" else PHASES.get(name)
            self.phase = (label if isinstance(label, str) else name)[:64]
            self.current = self.total = None
            share = 0.0
        self.estimate = max(self.estimate, round(low + (high - low) * share, 1))
        self.events += 1
        self.updated = self.last_advance = self.wall()
        return True

    def snapshot(self):
        return dict(stage=self.stage, estimate_percent=self.estimate, phase=self.phase, current=self.current,
                    total=self.total, updated=self.updated, last_advance=self.last_advance)


def finishing(snapshot, wall=time.time):
    """The conversion has ended: raster validation and publication remain."""
    value = dict(stage="starting", estimate_percent=0.0, phase=None, current=None, total=None,
                 updated=None, last_advance=None)
    value.update(snapshot or {})
    value.update(stage="finishing", estimate_percent=max(float(value["estimate_percent"]), 95.0), updated=wall())
    return value
