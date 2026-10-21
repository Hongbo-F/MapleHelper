"""Where the mini-map's records come from: a run followed live, a recorded run replayed, or a route
planned offline (no game needed)."""

import json
import time
from collections.abc import Callable
from pathlib import Path

from .. import telemetry


class LiveSource:
    """The newest telemetry file in a folder, followed as it grows; a newer run's file replaces it. The
    file is read on every poll, but the folder looked through at most once every `rescan_s`: the web page
    polls 40 times a second, and a new run is still picked up within a second. A run's next part (its
    first record the session record that `continues` the file followed, telemetry.Recorder) isn't a new
    run: the rest of the part before is read, then the new part, and the window and the page go on with
    what they had (the tour's and the hunt loop's grounds and safe platforms are sent once a run, her
    trail and the log lines go back further)."""

    def __init__(self, folder: Path = telemetry.FOLDER, rescan_s: float = 1.0,
                 clock: Callable[[], float] = time.monotonic):
        self.folder, self.rescan_s, self._clock = folder, rescan_s, clock
        self.path: Path | None = None
        self._follower: telemetry.Follower | None = None
        self._scanned: float | None = None  # when the folder was last looked through

    def poll(self) -> tuple[bool, list[dict]]:
        """(whether a new run started, the records since the last poll)."""
        now = self._clock()
        if self._scanned is None or now - self._scanned >= self.rescan_s:
            self._scanned = now
            newest = telemetry.newest(self.folder)
            if newest is not None and newest != self.path:
                follower = telemetry.Follower(newest)
                records = follower.poll()
                if self._follower is None or (records and not self._continues(records[0])):
                    self.path, self._follower = newest, follower
                    return True, records
                if records:  # the run's next part: the rest of this one, then on into it
                    rest = self._follower.poll()
                    self.path, self._follower = newest, follower
                    return False, rest + records
                # nothing in it yet (a part is made a moment before its first line): the next scan sees
        return False, (self._follower.poll() if self._follower else [])

    def _continues(self, first: dict) -> bool:
        """Whether a file starting with record `first` is the next part of the one followed."""
        return first.get("topic") == "session" and self.path is not None and first.get("continues") == self.path.name


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
