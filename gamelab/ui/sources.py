"""Where the mini-map's records come from: a run followed live, a recorded run replayed, or a route
planned offline (no game needed)."""

import json
import time
from collections.abc import Callable
from pathlib import Path

from .. import telemetry


class LiveSource:
    """The newest telemetry file in a folder, followed as it grows; a newer run's file replaces it."""

    def __init__(self, folder: Path = telemetry.FOLDER):
        self.folder = folder
        self.path: Path | None = None
        self._follower: telemetry.Follower | None = None

    def poll(self) -> tuple[bool, list[dict]]:
        """(whether a new run started, the records since the last poll)."""
        newest = telemetry.newest(self.folder)
        fresh = newest is not None and newest != self.path
        if fresh:
            self.path, self._follower = newest, telemetry.Follower(newest)
        return fresh, (self._follower.poll() if self._follower else [])


class ReplaySource:
    """Recorded records played back at their own pace (times `speed`)."""

    def __init__(self, records: list[dict], speed: float = 1.0, clock: Callable[[], float] | None = None):
        self.records = sorted(records, key=lambda r: r.get("t", 0.0))
        self.speed, self.paused = speed, False
        self._clock = clock or time.monotonic
        self.t0 = self.records[0].get("t", 0.0) if self.records else 0.0
        self.pos = 0.0          # seconds into the recording
        self._last = self._clock()
        self._next = 0

    def poll(self) -> tuple[bool, list[dict]]:
        now = self._clock()
        if not self.paused:
            self.pos += (now - self._last) * self.speed
        self._last = now
        out = []
        while self._next < len(self.records) and self.records[self._next].get("t", 0.0) - self.t0 <= self.pos:
            out.append(self.records[self._next])
            self._next += 1
        return False, out

    def seek(self, seconds: float) -> list[dict]:
        """Jump `seconds` (back or on); returns every record up to the new position (to rebuild the state)."""
        self.pos = max(0.0, self.pos + seconds)
        self._next = sum(1 for r in self.records if r.get("t", 0.0) - self.t0 <= self.pos)
        return self.records[:self._next]


def load(path: Path, map_id: int | None = None) -> list[dict]:
    """A recorded run's records: a telemetry file (.jsonl), or an older sweep track (JSON with "track":
    [[t, x, y, platform], ...], her positions only)."""
    if path.suffix == ".jsonl":
        return telemetry.read(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    track = data.get("track", data) if isinstance(data, dict) else data
    out = [{"topic": "session", "t": track[0][0] if track else 0.0, "map": map_id, "mode": "replay"}]
    for t, x, y, platform in track:
        out.append({"topic": "scene", "t": t, "her": {"x": x, "y": y, "platform": platform}})
    return out


def planned(model, src: int, x: float, dst: int) -> list[dict]:
    """The route from (platform src, x) to platform dst and its first chain, as the navigator would plan
    them (no game needed): records for a still picture."""
    from ..motion.flow import Body, Flow, chain_record
    from ..motion.route import Planner, route_record
    now = time.time()
    path = Planner(model).plan(src, x, dst) or []
    out = [{"topic": "session", "t": now, "map": model.id, "mode": "plan"},
           {"topic": "scene", "t": now, "her": {"x": x, "y": model.platform(src).y_at(x), "platform": src}},
           {"topic": "route", "t": now, **route_record(dst, path)}]
    if path:
        names = {k: k for k in ("left", "right", "jump", "up", "down")}  # no keys are sent: any names do
        legs, _, why = Flow(model, names).chain(path, Body(0.0, x, model.platform(src)))
        if legs:
            out.append({"topic": "chain", "t": now, **chain_record(legs, why)})
    out.append({"topic": "log", "t": now, "text": f"route {src} -> {dst}: " + (
        ", ".join(f"{m.kind}->{m.dst}" for m in path) if path else "none")})
    return out
