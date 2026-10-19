"""The mini-map's picture of the run: the latest record of each telemetry topic, her trail, and recent
log lines (events show as log lines too). Records are plain dicts (gamelab.telemetry)."""

from collections import deque
from dataclasses import dataclass, field

TRAIL = 240   # her last positions drawn as a trail
LOG_LINES = 9
LATEST = ("route", "chain", "monsters", "stance", "drops")  # topics kept as their latest record


@dataclass
class State:
    t: float = 0.0
    map: int | None = None
    session: dict = field(default_factory=dict)
    scene: dict | None = None      # her, hp, mp, monsters, loot (map units)
    route: dict | None = None      # the planned moves to a platform
    chain: dict | None = None      # the moves being carried out in one key plan, predicted
    monsters: dict | None = None   # monsters as the planner predicts them
    stance: dict | None = None     # the fight's last decision
    drops: dict | None = None      # drops remembered for later
    tour: dict | None = None       # the sweep's order, current stop and spot
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL))
    logs: deque = field(default_factory=lambda: deque(maxlen=LOG_LINES))

    def apply(self, record: dict) -> None:
        topic = record.get("topic")
        self.t = record.get("t", self.t)
        if topic in LATEST:
            setattr(self, topic, record)
        elif topic == "session":
            self.session = record
            self.map = record.get("map", self.map)
        elif topic == "scene":
            self.scene = record
            her = record.get("her") or {}
            if her.get("x") is not None:
                self.trail.append((her["x"], her["y"]))
        elif topic == "glance":
            if record.get("x") is not None:
                self.trail.append((record["x"], record["y"]))
        elif topic == "tour":
            self.tour = {**(self.tour or {}), **record}  # the sweep's order, then each stop and its spot
        elif topic == "trip":
            self.route = self.chain = None  # arrived, or gave up: the route is done with
        elif topic == "event":
            self.logs.append(record.get("text", record.get("kind", "")))
        elif topic == "log":
            self.logs.append(record.get("text", ""))

    def her(self) -> dict | None:
        """Her position: the last look's, moved on by any glance since."""
        her = (self.scene or {}).get("her")
        if not her or her.get("x") is None:
            return None
        if self.trail:
            x, y = self.trail[-1]
            her = {**her, "x": x, "y": y}
        return her
